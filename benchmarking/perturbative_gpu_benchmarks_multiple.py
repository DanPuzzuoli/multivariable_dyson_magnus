###############################################################################
# Run odeint_gpu_parallel_single.py over a list of num_input values, run on GPU
###############################################################################

import os
import subprocess

path_to_root = "/u/dpuzzuoli/multivariable_benchmarks"

# root string for save file
file_root = f"{path_to_root}/perturbative_gpu_benchmarks/"

# bash script file for job
script_file = f"{path_to_root}/perturbative_gpu_benchmarks_single.sh"

if __name__ == "__main__":

    num_inputs = 7000
    for expansion_method in ["dyson"]:#["dyson", "magnus"]:
        for chebyshev_order in [0]:#[0, 1, 2]:
            for expansion_order in [3]:#[2, 3, 4, 5]:
                for n_steps in [10000]:#[10000, 20000, 30000, 40000, 50000]:
                    file_name = f"{file_root}{expansion_method}_{chebyshev_order}_{expansion_order}_{n_steps}.json"
                    #print(file_name)
                    #print(f"{file_name[:-5]}_log.txt")
                    if os.path.isfile(file_name):
                        print(f"file {file_name} already exists, skipping job")
                    else:
                        bash_command = f"jbsub -e {file_name[:-5]}_log.txt -cores 8+1 -q x86_6h -require 'a100_80gb' -mem 160G sh {script_file} -n {num_inputs} -f {file_root} -e {expansion_method} -c {chebyshev_order} -o {expansion_order} -s {n_steps}"
                        subprocess.run(bash_command, shell=True)
