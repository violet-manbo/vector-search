#!/bin/bash

# =========================================
# 配置
# =========================================
BASE_DIR=~/vector_search
PYTHON_SCRIPT="$BASE_DIR/baseline/pilot/script/bench_2.py"
TOP_K=10
ALGORITHM="nsg"
N_QUERIES=1024
BASE_OUTPUT_DIR=$BASE_DIR/result/pilotann

mkdir -p "${BASE_OUTPUT_DIR}"

# checkpoint 列表（不同数据集）
CHECKPOINTS=( "deep10m.ckpt" "sift.ckpt" "gist.ckpt" "deep1m.ckpt" "mnist.ckpt")

# ef_search 多段区间

RANGES=( "30:80:2" )

# =========================================
# 函数：运行 benchmark
# =========================================
run_bench() {
    local ckpt=$1
    local dataset=${ckpt%%.*}          # 去掉 .ckpt 获取 dataset 名
    local output_dir="${BASE_OUTPUT_DIR}/${dataset}"

    mkdir -p "${output_dir}"

    # 释放 GPU 内存
    python3 -c "
import torch, time
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
time.sleep(10)
"

    # 循环 ef_search 参数
    for range_str in "${RANGES[@]}"; do
        IFS=':' read -r start end step <<< "${range_str}"
        for ef in $(seq "${start}" "${step}" "${end}"); do
            output_file="${output_dir}/runlog_${ef}.log"
            echo "Running checkpoint=${ckpt} ef_search=${ef} ..."
            python3 "$PYTHON_SCRIPT" \
                --checkpoint "${ckpt}" \
                --top_k "${TOP_K}" \
                --algorithm "${ALGORITHM}" \
                --n_queries "${N_QUERIES}" \
                --ef_search "[${ef}]" \
                > "${output_file}" 2>&1
            echo "Saved results to ${output_file}"
        done
    done
}

# =========================================
# 批量处理所有 checkpoint
# =========================================
for ckpt in "${CHECKPOINTS[@]}"; do
    run_bench "${ckpt}"
done

echo "All searches finished."
