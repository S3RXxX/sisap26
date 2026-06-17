# Ketchup Needs Nuggets' SISAP 2026 Indexing Challenge Solution

This project implements and evaluates nearest neighbor search algorithms for high-dimensional datasets for the [SISAP 2026 Indexing Challenge](https://sisap-challenges.github.io/2026/index.html).

It supports two tasks (maybe):

* **Task 1**: K-nearest neighbor graph (a.k.a. metric self-join)
* **Task 3**: Indexing very sparse high-dimensional vectors

---

## Project Structure


### Data Format
algo: Name of the algorithm (string).

task: Name of the task (task1).

buildtime: Index construction time in seconds (float). For task 1, also include data loading/preprocessing.

querytime: Total search time in seconds (float).

params: A string describing the parameters (parameters used in pipnn, but we already tuned them -no time, read the paper for more info-).

---

## Requirements
A folder data with the datasets that you want to use.
---

## Parameters Tuning
We don't have any parameter to tune.

---


## Evaluation


---




## Additional Notes
You need to indicate the input and the config from the input in config_pipnn.json. To execute the code run run_sisap2026.py as in entrypoint.sh


