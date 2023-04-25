# parse args
while getopts n:f:p:e: flag
do
    case "${flag}" in
        n) n=${OPTARG};;
        f) f=${OPTARG};;
        p) p=${OPTARG};;
        e) e=${OPTARG};;
    esac
done

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate devEnv310

# run python benchmarking script
python odeint_cpu_benchmarks_single.py --num_inputs $n --file_root $f --num_processes $p --tol_exp $e --compare_states
