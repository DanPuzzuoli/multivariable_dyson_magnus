# parse args
while getopts n:r: flag
do
    case "${flag}" in
        n) n=${OPTARG};;
        r) r=${OPTARG};;
    esac
done

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate devEnv310

# run python benchmarking script
python odeint_gpu_parallel_single.py --num_inputs $n --file_root $r