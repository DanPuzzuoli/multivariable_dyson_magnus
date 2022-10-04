# Paper supplement: Algorithms and software for computing and utilizing the Dyson series and Magnus expansion

This repository is a supplement to the paper in the above title, containing notebooks and code for generating the plots in the paper.

## Running the benchmarking

The paper contains a number of figures displaying relationships between speed and solver configurations, the data for these figures can be generated from this codebase, specifically using the `benchmarking` folder. Below are instructions to generate this data.

If running on a server cluster that has an LSF job queue, then you can use [part 1](#part-1-multiple-node-benchmarking) to run multiple simulations at once on different nodes, otherwise skip to [part 2](#part-2-running-benchmark-on-a-single-node) -- you will have to manually rerun the simulations to observe efficacy on different hardware.

### Part 1: Multiple Node Benchmarking

1. Edit the marked paths to your personal folders, on lines `10`, `12`, and `34` in `benchmarks/multi_node_benchmark_lsf.py`. These should point to the error path, the path to store the data output, and the submission bash script (`single_node_bash_runner.sh`), respectively

2. Edit the marked paths to the main python benchmarking file `single_node_benchmark.py` on line `97` of `single_node_bash_runner.sh`

3. No changes need to be made to `single_node_benchmark.py`, so `multi_node_benchmark_lsf.py` can be run to generate all the required data to make the plots

### Part 2: Running benchmark on a single node

For an individual execution of the benchmarking code, the file `single_node_benchmark.py` should be utilized. This file can be run as is without edits, with various default arguments for the possible parameters to examine.
