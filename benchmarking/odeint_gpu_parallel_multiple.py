###############################################################################
# Run odeint_gpu_parallel_single.py over a list of num_input values, run on GPU
###############################################################################

import subprocess

path_to_root = ""

# root string for save file
file_root = f"{path_to_root}/benchmarks/odeint_gpu_parallel"

# bash script file for job
script_file = f"{path_to_root}/odeint_gpu_parallel_single.sh"

if __name__ == "__main__":

    for num_inputs in [100, 1000, 2000, 3000, 4000, 5000, 6000]:
        bash_command = f"jbsub -cores 8+1 -q x86_6h -require 'a100_80gb' sh {script_file} -n {num_inputs} -r {file_root}"
        subprocess.run(bash_command, shell=True)