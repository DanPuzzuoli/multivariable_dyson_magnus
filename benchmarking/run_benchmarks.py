import subprocess
import os


def run_job(cores, gpu, vmap, solver, n_inputs, database_path, proj, dim):

    label = f"{'gpu' if gpu else cores}{solver}{'vmap' if vmap else ''}{n_inputs}"

    err_path = (
        f"/u/brosand/projects/danDynamics/multivariable_dyson_magnus/{proj}/{label}"
    )
    newpath = f"/u/brosand/projects/danDynamics/multivariable_dyson_magnus/{proj}"
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

    bashCommand = f"jbsub -e {err_path}.err -o {err_path}.out{gpu_require} -cores {core_string} -mem 160G -q x86_24h -proj {proj} /u/brosand/projects/danDynamics/multivariable_dyson_magnus/benchmarking_scripts/sim_bash.sh --cpus {core_count} --solver {solver} --file_name gpu0v{vmap_string} --n_inputs {n_inputs} --sql {database_path} --dim {dim}"
    subprocess.run(bashCommand, shell=True)


inputs = [1, 50, 100]
solvers = ["dense", "dyson", "magnus"]
vmap = True
dim = 5

# Run CPU simulations
gpus = [False]
cores = [1, 64]
proj = "cpu_data.sqlite"
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
proj = "gpu_data.sqlite"
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
