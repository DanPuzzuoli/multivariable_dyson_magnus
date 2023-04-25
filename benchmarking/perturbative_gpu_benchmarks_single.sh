# parse args
while getopts n:f:e:c:o:s: flag
do
    case "${flag}" in
        n) n=${OPTARG};;
        f) f=${OPTARG};;
        e) e=${OPTARG};;
        c) c=${OPTARG};;
	o) o=${OPTARG};;
	s) s=${OPTARG};;
    esac
done

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate devEnv310

# run python benchmarking script
python perturbative_gpu_benchmarks_single.py --num_inputs $n --file_root $f --expansion_method $e --chebyshev_order $c --expansion_order $o --n_steps $s --compare_states
