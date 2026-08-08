# Ketchup Needs Nuggets' SISAP 2026 Indexing Challenge Solution

This project implements and evaluates nearest neighbor search algorithms for high-dimensional datasets for the [SISAP 2026 Indexing Challenge](https://sisap-challenges.github.io/2026/index.html).

It supports one task:

* **Task 1**: K-nearest neighbor graph (a.k.a. metric self-join)

---

## Prerequisites
You will need to build and execute a docker. For windows we recommend to install Docker Desktop and for linux docker.

## Additional Notes
To build our project:
- docker build -t pipnn-sisap .

To execute our solution with the best parameters found:
- docker run --rm -v ${PWD}/data:/data -e K=15 -e BEAM_WIDTH=32 -e ALPHA=1.2 -e LEAF_SIZE=512 -e MIN_LEAF_SIZE=32 -e FINAL_PRUNE=0 -e BACK_EDGE=0 pipnn-sisap

## Files
- pipnn_dot.hpp: Has our implementation of PIPNN.
- sisap_bench.cpp: Execute PIPNN for train and the different tests from SISAP 2026 Indexing Challenge.
- run_sisap2026.py: Compile and execute sisap_bench.cpp.
- sweep.py: Experimentation code that uses the docker.
- *.ipynb: Code used to create images.
