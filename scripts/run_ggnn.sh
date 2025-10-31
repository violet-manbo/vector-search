#!/bin/bash


# dir
BASE_DIR=~/vector_search
GGNN_DIR=$BASE_DIR/baseline/ggnn/build


# dataset degree metric
declare -A DATASETS
DATASETS[gist]="128 l2"
DATASETS[sift]="32 l2"
DATASETS[nytimes]="64 cos"
DATASETS[deep]="64 cos"
DATASETS[glove]="96 cos"
DATASETS[mnist]="128 l2"
DATASETS[deep100m]="32 l2"
DATASETS[deep1m]="32 l2"

# topk
K=10

# set iter
NP_RANGE=$(seq 30 2 60)


run_ggnn() {
    local dataset=$1
    local d=$2
    local metric=$3

    local base_file=$BASE_DIR/dataset/$dataset/base.fvecs
    local query_file=$BASE_DIR/dataset/$dataset/query.fvecs
    local gt_file=$BASE_DIR/dataset/$dataset/groundtruth.ivecs
    local graph_dir=$BASE_DIR/graph/ggnn/$dataset

    local output_dir=$BASE_DIR/result/ggnn/topk/$dataset
    local recall_dir=$BASE_DIR/result/ggnn/recall/$dataset
    

    mkdir -p $output_dir
    mkdir -p $recall_dir
    mkdir -p $BASE_DIR/result/ggnn/runlog/$dataset


    for i in $NP_RANGE; do
        echo "Running GGNN on $dataset with nprobe=$i ..."
        $GGNN_DIR/ggnn_main_gpu_data query $base_file $query_file $graph_dir $d $K $i $metric $output_dir/result_$i.txt > $BASE_DIR/result/ggnn/runlog/$dataset/runlog_$i.log 2>&1

        cd $BASE_DIR
        python ./Scripts/cal_recall_ggnn.py --gt $gt_file --ret $output_dir/result_$i.txt --k $K --out $recall_dir/recall_$i.txt
        cd $GGNN_DIR
    done
}


for dataset in "${!DATASETS[@]}"; do
    params=(${DATASETS[$dataset]})
    run_ggnn $dataset ${params[0]} ${params[1]}
done

echo "All GGNN experiments finished."
