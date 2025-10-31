#!/bin/bash


# dir

BASE_DIR=~/vector_search
SONG_DIR=$BASE_DIR/baseline/song

# dataset
DATASETS=("nytimes" "sift" "gist" "glove" "deep1m" "deep10m")  # 可以按实际添加

# dataset：k n dim metric
declare -A DATASET_PARAMS
DATASET_PARAMS[nytimes]="10 290000 256 cos"
DATASET_PARAMS[sift]="10 1000000 128 l2"
DATASET_PARAMS[gist]="10 1000000 960 l2"
DATASET_PARAMS[glove]="10 1184514 200 cos"
DATASET_PARAMS[deep1m]="10 1000000 96 l2"
DATASET_PARAMS[deep10m]="10 10000000 96 l2"


# run
run_song() {
    local dataset=$1
    local params=($2)   # d n k metric

    local k=${params[0]}
    local n=${params[1]}
    local dim=${params[2]}
    local metric=${params[3]}

    local query_file=$BASE_DIR/dataset/$dataset/query.libsvm
    local gt_file=$BASE_DIR/dataset/$dataset/groundtruth.ivecs
    local output_dir=$BASE_DIR/result/song/$dataset/output
    local recall_dir=$BASE_DIR/result/song/$dataset/recall

    mkdir -p $output_dir
    mkdir -p $recall_dir

    echo "Running Song on $dataset ..."
    $SONG_DIR/song test 0 $query_file $d $n $d $k $metric > $output_dir/output.txt 2>&1

    cd $BASE_DIR
    python ./Scripts/cal_recall_song.py --gt $gt_file --ret $output_dir/output.txt --k $k --out $recall_dir/recall.txt
    cd $SONG_DIR
}


for dataset in "${DATASETS[@]}"; do
    run_song $dataset "${DATASET_PARAMS[$dataset]}"
done

echo "All experiments finished."
