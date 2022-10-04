import subprocess
import os


def run_job(cores, gpu, vmap, solver, n_inputs, database_path, proj, dim, test=False):

    label = f"{'gpu' if gpu else cores}{solver}{'vmap' if vmap else ''}{n_inputs}"

    err_path = (
        f"<path_to_root_folder>/benchmarking/{proj}/{label}"
    )
    newpath = f"<path_to_root_folder>/benchmarking/{proj}"
    if not os.path.exists(newpath):
        os.mkdir(newpath)

    if gpu:
        gpu_require = ' -require "a100_80gb"'
        core_string = "8+1"
        core_count = 0
    else:
        gpu_require = ""
        core_string = cores
        core_count = cores

    if vmap:
        vmap_string = " --vmap"
    else:
        vmap_string = ""

    if test:
        test_arg = "--test"
    else:
        test_arg = ""
    bashCommand = f"jbsub -e {err_path}.err -o {err_path}.out{gpu_require} -cores {core_string} -mem 160G -q x86_24h -proj {proj} <path_to_root_folder>/benchmarking/single_node_bash_runner.sh --cpus {core_count} --solver {solver} --file_name gpu0v{vmap_string} --n_inputs {n_inputs} --sql {database_path} --dim {dim} {test_arg}"
    subprocess.run(bashCommand, shell=True)


inputs = [1, 50, 100]
solvers = ["dense", "dyson", "magnus"]
vmap = True
dim = 5

# Run CPU simulations
gpus = [False]
cores = [1, 64]
proj = "cpu_data"
database_path = f"{proj}.sqlite"

for input in inputs:
    for solver in solvers:
        for core in cores:
            run_job(
                core,
                gpu=False,
                vmap=vmap,
                solver=solver,
                n_inputs=input,
                database_path=database_path,
                proj=proj,
                dim=dim,
            )

# Run GPU simulations
proj = "gpu_data"
database_path = f"{proj}.sqlite"

for input in inputs:
    for solver in solvers:
        run_job(
            1,
            gpu=True,
            vmap=vmap,
            solver=solver,
            n_inputs=input,
            database_path=database_path,
            proj=proj,
            dim=dim,
        )
