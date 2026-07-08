# from huggingface_hub import hf_hub_download

# # Download a specific dataset
# file_path = hf_hub_download(
#     repo_id="SISAP-Challenges/SISAP2026",
#     filename="wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5",
#     repo_type="dataset"
# )

# # Download a config file
# config_path = hf_hub_download(
#     repo_id="SISAP-Challenges/SISAP2026",
#     filename="wikipedia-small/config.json",
#     repo_type="dataset"
# )

# print(f"Data path: {file_path}")
# print(f"Config path: {config_path}")


#!/usr/bin/env python3

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "SISAP-Challenges/SISAP2026"
DATASET_DIR = "wikipedia"

FILES = [
    f"{DATASET_DIR}/benchmark-dev-wikipedia-bge-m3.h5",
    f"{DATASET_DIR}/config.json",
]


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print(f"Downloading dataset to: {data_dir.resolve()}")

    for filename in FILES:
        print(f"Downloading {filename}...")

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir,
            local_dir_use_symlinks=False,  # copies files into data/
        )

        print(f"✓ Saved to {local_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()