#!/bin/bash

# dir
BASE_DIR=~/vector_search
GANNS_DIR=$BASE_DIR/baseline/GANNS

# dataset: exe dim metric
declare -A DATASETS
DATASETS[gist]="query_960_l2 960 l2"
DATASETS[deep10m]="query_96_l2 96 l2"
DATASETS[sift]="query_128_l2 128 l2"
DATASETS[nytimes]="query_256_cos 256 cos"
DATASETS[glove]="query_200_cos 200 cos"
DATASETS[mnist]="query_784_l2 784 l2"
DATASETS[deep1m]="query_96_l2 96 l2"

# set iter
NP_RANGE=$(seq 30 5 60)

# run
run_ganns() {
    local dataset=$1
    local query_bin=$2
    local d=$3
    local metric=$4

    local base_file=$BASE_DIR/dataset/$dataset/base.fvecs
    local query_file=$BASE_DIR/dataset/$dataset/query.fvecs
    local gt_file=$BASE_DIR/dataset/$dataset/groundtruth.ivecs
    local graph=$BASE_DIR/graph/ganns/${dataset}.nsw

    local output_dir=$BASE_DIR/result/ganns/$dataset
    mkdir -p $output_dir

    for i in $NP_RANGE; do
        echo "Running GANNS on $dataset with nprobe=$i ..."
        $GANNS_DIR/$query_bin $base_file $query_file nsw $graph $gt_file $i 10 > $output_dir/runlog_$i.log 2>&1
    done
}

for dataset in "${!DATASETS[@]}"; do
    params=(${DATASETS[$dataset]})
    run_ganns $dataset ${params[0]} ${params[1]} ${params[2]}
done

echo "All GANNS experiments finished."
