#!/bin/bash


# dir
BASE_DIR=~/vector_search
BANG_DIR=$BANG_DIR/baseline/bang/build


#dataset degree metric
declare -A DATASETS
DATASETS[gist]="1000 l2" 
DATASETS[sift]="10000 l2"
DATASETS[nytimes]="10000 cos"
DATASETS[deep]="10000 l2"
DATASETS[glove]="10000 cos"
DATASETS[mnist]="10000 l2"
DATASETS[deep100m]="10000 l2"
DATASETS[deep1m]="10000 l2"

# topk
K=10





# run

run_ggnn() {
    local dataset=$1
    local n_queries=$2
    local dist_fn=$3
    local graph_prefix=$BASE_DIR/graph/bang/${dataset}
    local base_file=$BASE_DIR/dataset/$dataset/base.bin
    local query_file=$BASE_DIR/dataset/$dataset/query.bin
    local gt_file=$BASE_DIR/dataset/$dataset/groundtruth.bin
    

    local runlog_dir=$BASE_DIR/result/bang/$dataset
    

    mkdir -p $runlog_dir

    $BANG_DIR/bang_search $graph_prefix $query_file $gt_file $n_queries $K float $dist_fn > $runlog_dir/runlog.txt 2>&1
}



for dataset in "${!DATASETS[@]}"; do
    params=(${DATASETS[$dataset]})
    run_ggnn $dataset ${params[0]} ${params[1]}
done

echo "All GGNN experiments finished."
