###############################################################################
# Run odeint_gpu_parallel_single.py over a list of num_input values, run on GPU
###############################################################################

import subprocess

path_to_root = "/u/dpuzzuoli/multivariable_benchmarks"

# root string for save file
file_root = f"{path_to_root}/odeint_cpu_benchmarks/tol"

# bash script file for job
script_file = f"{path_to_root}/odeint_cpu_benchmarks_single.sh"

if __name__ == "__main__":

    num_inputs = 7000
    for tol_exp in [6, 7, 8, 9, 10, 11, 12, 13]:
        bash_command = f"jbsub -e error_{tol_exp}.txt -cores 100 -q x86_6h -mem 160G sh {script_file} -n {num_inputs} -f {file_root} -p 100 -e {tol_exp}"
        subprocess.run(bash_command, shell=True)
