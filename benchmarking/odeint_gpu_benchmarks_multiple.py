###############################################################################
# Run odeint_gpu_parallel_single.py over a list of num_input values, run on GPU
###############################################################################

import subprocess

path_to_root = "/u/dpuzzuoli/multivariable_benchmarks"

# root string for save file
file_root = f"{path_to_root}/odeint_gpu_benchmarks/tol"

# bash script file for job
script_file = f"{path_to_root}/odeint_gpu_benchmarks_single.sh"

if __name__ == "__main__":

    # first run jbsub -e odeint_gpu_benchmarks/tol_error_14.txt -cores 8+1 -q x86_6h -require 'a100_80gb' -mem 160G sh odeint_gpu_benchmarks_single.sh -n 7000 -p 7000 -e 14
    # with odeint_gpu_benchmarks_single.sh modified to save states, to generate states for benchmarking distance
    num_inputs = 7000
    for tol_exp in [6, 7, 8, 9, 10, 11, 12, 13]:
        bash_command = f"jbsub -e {file_root}_error_{tol_exp}.txt -cores 8+1 -q x86_6h -require 'a100_80gb' -mem 160G sh {script_file} -n {num_inputs} -f {file_root} -p {num_inputs} -e {tol_exp}"
        subprocess.run(bash_command, shell=True)
