##############################################################
# Script for testing parallelization capacity of odeint on GPU
##############################################################

import json
import argparse

def odeint_benchmark_run(file_name, num_inputs, num_processes, tol, save_state_file=None, compare_to_benchmark=None):

    import jax
    jax.config.update("jax_enable_x64", True)

    from qiskit_dynamics.array import Array
    Array.set_default_backend("jax")

    # tell JAX we are using GPU
    jax.config.update('jax_platform_name', 'gpu')
    from benchmark_functions import odeint_benchmark

    
    metrics = odeint_benchmark(
        tol=tol,
        evaluation_mode="dense",
        num_inputs=num_inputs,
        num_processes=num_processes,
        parallel_mode="vmap",
        save_state_file=save_state_file,
        compare_to_benchmark=compare_to_benchmark,
        benchmark_gradient=True
    )

    json.dump(metrics, open(file_name, 'w' ) )
    jax.profiler.save_device_memory_profile(f"{file_name[:-5]}.prof")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_root", default="odeint_gpu_benchmarks/tol")
    parser.add_argument("--num_inputs", default=7000)
    parser.add_argument("--num_processes", default=100)
    parser.add_argument("--tol_exp", default=8)
    parser.add_argument("--save_states", action='store_true')
    parser.add_argument("--compare_states", action='store_true')
    args = parser.parse_args()

    compare_to_benchmark = None
    if args.compare_states:
        compare_to_benchmark = f"{args.file_root}_14_states.npy"


    odeint_benchmark_run(
        file_name=f"{args.file_root}_{str(args.tol_exp)}.json", 
        num_inputs=int(args.num_inputs),
        num_processes=int(args.num_processes),
        tol=10**(-int(args.tol_exp)),
        save_state_file=None if not bool(args.save_states) else f"{args.file_root}_{str(args.tol_exp)}_states.npy",
        compare_to_benchmark=compare_to_benchmark
    )
