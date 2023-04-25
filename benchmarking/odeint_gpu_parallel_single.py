##############################################################
# Script for testing parallelization capacity of odeint on GPU
##############################################################

import os
import json
import argparse

import jax
jax.config.update("jax_enable_x64", True)

from qiskit_dynamics.array import Array
Array.set_default_backend("jax")

# tell JAX we are using CPU
jax.config.update('jax_platform_name', 'cpu')

from benchmark_functions import odeint_benchmark

def odeint_benchmark_parallel(file_name, num_inputs):

    if os.path.isfile(file_name):
        print("file already exists, skipping")
        # exit if file already exists and overwrite is False
        return
    
    metrics = odeint_benchmark(
        tol=1e-10,
        evaluation_mode="dense",
        num_inputs=num_inputs,
        num_processes=num_inputs,
        parallel_mode="vmap"
    )

    json.dump(metrics, open(file_name, 'w' ) )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_root", default="benchmarks/odeint_gpu_parallel")
    parser.add_argument("--num_inputs", default=10)
    args = parser.parse_args()

    num_inputs = int(args.num_inputs)
    file_name = f"{args.file_root}_{str(num_inputs)}.json" 

    odeint_benchmark_parallel(file_name, num_inputs)