#!/bin/bash


# dir

BASE_DIR=~/vector_search
BUILD_DIR=$BASE_DIR/baseline/cagra/build

#topk
K=10

#set  iter
ITER_RANGE=($(seq 30 2 60))

# dataset
DATASETS=(deep10m glove nytimes sift mnist deep100m deep1m gist)


# run

run_cagra() {
    local dataset=$1

    local query_file=$BASE_DIR/dataset/$dataset/query.fvecs
    local gt_file=$BASE_DIR/dataset/$dataset/groundtruth.ivecs
    local graph=$BASE_DIR/graph/cagra/$dataset
    local output_dir=$BUILD_DIR/topk/$dataset
    local recall_dir=$BUILD_DIR/recall/$dataset
    local log_dir=$BUILD_DIR/runlog/$dataset

    mkdir -p $output_dir
    mkdir -p $recall_dir
    mkdir -p $log_dir

    for iter in "${ITER_RANGE[@]}"; do
        echo "Running CAGRA on $dataset with iter=$iter ..."

        $BUILD_DIR/my_cagra search $graph $query_file $iter \
            $output_dir/output_${iter} 1 128 8 > $log_dir/${dataset}_${iter}.log 2>&1

        python $BASE_DIR/Scripts/cal_recall_cagra.py \
            --gt $gt_file \
            --ret $output_dir/output_${iter} \
            --k $K \
            --out $recall_dir/recall_${iter}.txt
    done
}




for dataset in "${DATASETS[@]}"; do
    run_cagra $dataset
done

echo "All CAGRA experiments finished."
