
import json
import argparse

def perturbative_benchmark_run(file_name, num_inputs, expansion_method,  chebyshev_order, expansion_order, n_steps, save_state_file=None, compare_to_benchmark=None):
    import os
    #os.environ["JAX_JIT_PJIT_API_MERGE"] = "0"
    import jax
    jax.config.update("jax_enable_x64", True)

    from qiskit_dynamics.array import Array
    Array.set_default_backend("jax")
    print(jax.devices())
    # tell JAX we are using GPU
    jax.config.update('jax_platform_name', 'gpu')
    from benchmark_functions import perturbative_benchmark

    
    metrics = perturbative_benchmark(
        expansion_method=expansion_method,
        chebyshev_order=chebyshev_order,
        expansion_order=expansion_order,
        n_steps=n_steps,
        num_inputs=num_inputs,
        num_processes=None,
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
    parser.add_argument("--file_root", default="perturbative_gpu_benchmarks/")
    parser.add_argument("--num_inputs", default=7000)
    parser.add_argument("--expansion_method", default="dyson")
    parser.add_argument("--chebyshev_order", default=0)
    parser.add_argument("--expansion_order", default=3)
    parser.add_argument("--n_steps", default=10000)
    parser.add_argument("--save_states", action='store_true')
    parser.add_argument("--compare_states", action='store_true')
    args = parser.parse_args()

    compare_to_benchmark = None
    if args.compare_states:
        # compare to states computed using odeint with 1e-14 tolerance
        compare_to_benchmark = f"{args.file_root}tol_14_states.npy"

    expansion_method = str(args.expansion_method)
    chebyshev_order = int(args.chebyshev_order)
    expansion_order = int(args.expansion_order)
    n_steps = int(args.n_steps)


    perturbative_benchmark_run(
        file_name=f"{args.file_root}{expansion_method}_{chebyshev_order}_{expansion_order}_{n_steps}.json", 
        num_inputs=int(args.num_inputs),
        expansion_method=expansion_method,
        chebyshev_order=chebyshev_order,
        expansion_order=expansion_order,
        n_steps=n_steps,
        save_state_file=None if not bool(args.save_states) else f"{args.file_root}{expansion_method}_{chebyshev_order}_{expansion_order}_{n_steps}_states.npy",
        compare_to_benchmark=compare_to_benchmark
    )
