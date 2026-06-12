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
- docker run --rm -v ${PWD}/data:/data -e K=15 -e RAND=1 -e BEAM_WIDTH=32 -e MAX_DEGREE=64 -e MIN_LEAF_SIZE=16 -e LEAF_SIZE=32 -e ALPHA=1.1 -e NUM_THREADS=0 -e NUM_THREADS=0 -e MEMORY_LIMIT_GB=16 -e COOCKED=0 pipnn-sisap
