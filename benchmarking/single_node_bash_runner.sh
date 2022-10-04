#!/bin/bash
# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# PIPESTATUS with a simple $?, but I don’t do that.
set -o errexit -o pipefail -o noclobber -o nounset

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

# OPTIONS=dfo:v
# LONGOPTS=debug,force,output:,verbose

OPTIONS=c:s:n:d:q:vrt
LONGOPTS=cpus:,solver:,sql:,file_name:,n_inputs:,dim:,vmap,norft,test,

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

c=- s=- n=- d=- v='' r='' t='' f=- q=-
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -t|--test)
            t=--test
            shift
            ;;
        -r|--norft)
            r=--norft
            shift
            ;;
        -v|--vmap)
            v=--vmap
            shift
            ;;
        -s|--solver)
            s="$2"
            shift 2
            ;;
        -q|--sql)
            q="$2"
            shift 2
            ;;
        -c|--cpus)
            c="$2"
            shift 2
            ;;
        -d|--dim)
            d="$2"
            shift 2
            ;;

        -n|--n_inputs)
            n="$2"
            shift 2
            ;;

        -f|--file_name)
            f="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

# handle non-option arguments
# if [[ $# -ne 1 ]]; then
#     echo "$0: A single input file is required."
#     exit 4
# fi


echo "cpus: $c, solver: $s, rft: "$r", vmap: "$v", test: "$t", output_name: "$f", sql_output: "$q", dim: $d"
cd /u/brosand/projects/danDynamics/multivariable_dyson_magnus
source /speech6/bedk3_nb/ECOC/anaconda3/bin/activate test
python /u/brosand/projects/danDynamics/multivariable_dyson_magnus/benchmarking/single_node_benchmark.py --output_name $f --n_inputs $n --solver $s --sql $q --dim $d --cpus $c $v $r $t
