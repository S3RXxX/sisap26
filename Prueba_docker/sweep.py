#!/usr/bin/env python3
"""
sweep.py — Barrido de hiperparámetros para PiPNN sobre Docker.

Ejecuta sisap_bench con todas las combinaciones de parámetros definidas en
PARAM_GRID, parsea el recall y el tiempo de cada ejecución y guarda los
resultados en results/sweep_results.csv + results/sweep_results.json.

Uso
───
    python sweep.py                     # usa los valores por defecto
    python sweep.py --data $(pwd)/data  # directorio con los datos
    python sweep.py --dry-run           # imprime los comandos sin ejecutar
    python sweep.py --jobs 2            # hasta 2 experimentos en paralelo

Añadir más parámetros al barrido
──────────────────────────────────
Solo hay que editar PARAM_GRID debajo. Cada clave es una variable de entorno
del contenedor y el valor es la lista de valores que se quieren probar.
Las combinaciones se generan con producto cartesiano.
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS A BARRER  ← edita aquí
# Cada lista contiene los valores que se quieren probar para ese parámetro.
# El producto cartesiano genera todos los experimentos.
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    # "BEAM_WIDTH":    [128, 256, 512],
    # "MAX_DEGREE":    [32, 64, 96],
    # "ALPHA":         [1.2, 1.4],
    "RAND":          [0, 1],
    "COOCKED":       [0,1],
    "SEED":          [42, 128, 5, 23, 47]
    # "LEAF_SIZE":   [256, 512],       # descomenta para añadir más dimensiones
    # "NUM_REPLICAS":[1, 2],
}

# Parámetros fijos (no se barren, solo se pasan como contexto)
FIXED_PARAMS = {
    "K":              15,
    "BEAM_WIDTH":     256,
    "MAX_DEGREE":     64,
    "ALPHA":          1.2,
    "LEAF_SIZE":      512,
    "MIN_LEAF_SIZE":  32,
    "K_ENTRY":        12,
    "ENTRY_SAMPLE":   3000,
    "HASH_BITS":      12,
    "RESERVOIR_CAP":  128,
    "NUM_REPLICAS":   1,
    "FINAL_PRUNE":    1,
    "BACK_EDGE":      1,
    "NUM_THREADS":    0,
    "MEMORY_LIMIT_GB": 0,
}

DOCKER_IMAGE = "pipnn-sisap"
RESULTS_DIR  = "results"

# ─────────────────────────────────────────────────────────────────────────────
# Parseo del output de sisap_bench
# ─────────────────────────────────────────────────────────────────────────────
# Líneas relevantes del output:
#   Build time : 12.34 s
#   Avg degree : 47.2
#   ── allknn (N queries, recall@K) ──
#     bw     recall@15   QPS
#     256    0.9821      4823
#   ── itest  ...
#   ── otest  ...

RE_BUILD_TIME  = re.compile(r"Build time\s*:\s*([\d.]+)\s*s")
RE_AVG_DEG     = re.compile(r"Avg degree\s*:\s*([\d.]+)")
RE_SECTION     = re.compile(r"[--]\s*(allknn|itest|otest)\s*\((\d+) queries")
RE_RESULT_ROW  = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")
RE_MAX_DEPTH = re.compile(r"[Mm]ax.?depth\s*[:\s]\s*(\d+)")


def parse_output(text: str) -> dict:
    """
    Extrae del stdout de sisap_bench:
      - build_time_s
      - avg_degree
      - por cada sección (allknn/itest/otest) y beam_width:
            recall, qps
    Devuelve un dict plano listo para escribir en CSV.
    """
    result = {}
    current_section = None

    m = RE_BUILD_TIME.search(text)
    result["build_time_s"] = float(m.group(1)) if m else None

    m = RE_AVG_DEG.search(text)
    result["avg_degree"] = float(m.group(1)) if m else None

    m = RE_MAX_DEPTH.search(text)
    result["max_depth"] = float(m.group(1)) if m else None

    for line in text.splitlines():
        ms = RE_SECTION.search(line)
        if ms:
            current_section = ms.group(1)   # allknn | itest | otest

        mr = RE_RESULT_ROW.match(line)
        if mr and current_section:
            bw     = int(mr.group(1))
            recall = float(mr.group(2))
            qps    = float(mr.group(3))
            # Guarda solo la fila cuyo bw coincide con el BEAM_WIDTH del experimento
            # (las otras filas son variantes internas del benchmark a BW/4, BW/2, BW*2)
            result.setdefault(f"{current_section}_rows", []).append(
                {"bw": bw, "recall": recall, "qps": qps}
            )

    # Aplana: para cada sección conserva todas las filas como columnas numeradas
    for section in ("allknn", "itest", "otest"):
        rows = result.pop(f"{section}_rows", [])
        for row in rows:
            prefix = f"{section}_bw{row['bw']}"
            result[f"{prefix}_recall"] = row["recall"]
            result[f"{prefix}_qps"]    = row["qps"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución de un experimento
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_id: int, params: dict, data_dir: str, dry_run: bool) -> dict:
    """
    Lanza un contenedor Docker con los parámetros dados.
    Devuelve un dict con todos los parámetros + métricas parseadas.
    """
    all_params = {**FIXED_PARAMS, **params}   # los del grid sobreescriben los fijos

    mem = all_params.get("MEMORY_LIMIT_GB", 0)
    mem_flag = [f"--memory={mem}g"] if mem and int(mem) > 0 else []

    env_flags = []
    for k, v in all_params.items():
        if k != "MEMORY_LIMIT_GB":
            env_flags += ["-e", f"{k}={v}"]

    cmd = (
        ["docker", "run", "--rm"]
        + mem_flag
        + ["-v", f"{data_dir}:/data"]
        + env_flags
        + [DOCKER_IMAGE]
    )

    label = "  ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n[{exp_id}] {label}")

    if dry_run:
        print("    CMD:", " ".join(cmd))
        return {"exp_id": exp_id, **params, "_dry_run": True}

    t_wall = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,    # 1 h máximo por experimento
        )
        wall_s = time.time() - t_wall

        if proc.returncode != 0:
            print(f"    ✗ Error (rc={proc.returncode})")
            print(proc.stderr[-500:] if proc.stderr else "")
            return {"exp_id": exp_id, **params, "error": proc.stderr[-200:]}

        metrics = parse_output(proc.stdout)
        metrics["wall_time_s"] = round(wall_s, 2)

        # Log de la fila principal (BEAM_WIDTH del experimento)
        bw = params.get("BEAM_WIDTH", all_params.get("BEAM_WIDTH", "?"))
        for section in ("allknn", "itest", "otest"):
            rec = metrics.get(f"{section}_bw{bw}_recall")
            qps = metrics.get(f"{section}_bw{bw}_qps")
            if rec is not None:
                print(f"    {section:8s}  recall={rec:.4f}  qps={qps:.0f}")
        if metrics.get("max_depth") is not None:
            print(f"    max_depth = {metrics['max_depth']}")
        if metrics.get("build_time_s"):
            print(f"    build={metrics['build_time_s']:.1f}s  "
                  f"wall={wall_s:.1f}s")

        return {"exp_id": exp_id, **params, **metrics}

    except subprocess.TimeoutExpired:
        print("    ✗ Timeout")
        return {"exp_id": exp_id, **params, "error": "timeout"}
    except Exception as exc:
        print(f"    ✗ {exc}")
        return {"exp_id": exp_id, **params, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Guardado de resultados
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON (completo)
    json_path = os.path.join(out_dir, f"sweep_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV (aplanado)
    csv_path = os.path.join(out_dir, f"sweep_{ts}.csv")
    all_keys = []
    for r in results:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados guardados en:\n  {csv_path}\n  {json_path}")
    return csv_path, json_path


def print_summary(results: list, bw_key: str):
    """Tabla resumen ordenada por itest recall descendente."""
    valid = [r for r in results if "error" not in r and not r.get("_dry_run")]
    if not valid:
        return

    # Columnas del grid + métricas clave
    grid_keys = list(PARAM_GRID.keys())
    metric_keys = [k for k in valid[0] if k.endswith("_recall") or k == "build_time_s"]

    # Ordena por itest recall del beam principal (si existe)
    itest_col = f"itest_{bw_key}_recall"
    valid.sort(key=lambda r: r.get(itest_col, 0), reverse=True)

    col_w = 10
    header = " ".join(f"{k:<{col_w}}" for k in grid_keys + metric_keys)
    print("\n" + "─" * len(header))
    print("RESUMEN (ordenado por itest recall desc)")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for r in valid:
        row = " ".join(
            f"{str(r.get(k, '')):<{col_w}}" for k in grid_keys + metric_keys
        )
        print(row)
    print("─" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Barrido de hiperparámetros PiPNN")
    parser.add_argument("--data",    default=f"{os.getcwd()}/data",
                        help="Directorio de datos montado en /data del contenedor")
    parser.add_argument("--image",   default=DOCKER_IMAGE,
                        help="Nombre de la imagen Docker")
    parser.add_argument("--out",     default=RESULTS_DIR,
                        help="Directorio donde guardar CSV y JSON")
    parser.add_argument("--jobs",    type=int, default=1,
                        help="Experimentos en paralelo (cuidado con la RAM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime los comandos sin ejecutar")
    args = parser.parse_args()

    # Genera el producto cartesiano de PARAM_GRID
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"Imagen Docker : {args.image}")
    print(f"Datos         : {args.data}")
    print(f"Experimentos  : {len(combos)}")
    print(f"Paralelismo   : {args.jobs}")
    print(f"Grid:\n" + "\n".join(f"  {k}: {v}" for k, v in PARAM_GRID.items()))

    # if not args.dry_run:
    #     # Comprueba que la imagen existe antes de empezar
    #     check = subprocess.run(
    #         ["docker", "image", "inspect", args.image],
    #         capture_output=True
    #     )
    #     if check.returncode != 0:
    #         print(f"\nError: imagen '{args.image}' no encontrada.")
    #         print(f"Constrúyela con:  docker build -t {args.image} .")
    #         sys.exit(1)

    results = []
    t_total = time.time()

    if args.jobs == 1:
        # Secuencial — más fácil de leer el output
        for i, combo in enumerate(combos, 1):
            if combo["RAND"] == 1 and combo["COOCKED"] == 1:
                continue
            r = run_experiment(i, combo, args.data, args.dry_run)
            results.append(r)
    else:
        # Paralelo con ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {
                ex.submit(run_experiment, i, combo, args.data, args.dry_run): i
                for i, combo in enumerate(combos, 1)
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    # Ordena por exp_id para reproducibilidad
    results.sort(key=lambda r: r.get("exp_id", 0))

    total_s = time.time() - t_total
    print(f"\nTiempo total: {total_s:.1f}s  ({total_s/60:.1f} min)")

    if not args.dry_run:
        # Determina el bw principal para el resumen
        sample_bw = combos[0].get("BEAM_WIDTH", FIXED_PARAMS.get("BEAM_WIDTH", 256))
        print_summary(results, f"bw{sample_bw}")
        save_results(results, args.out)


if __name__ == "__main__":
    main()
