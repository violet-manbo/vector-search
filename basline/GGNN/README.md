# SONG

## Overview 

GGNN performs nearest-neighbor computations on CUDA-capable GPUs. It is based on the method proposed in the paper [GGNN: Graph-based GPU Nearest Neighbor Search](https://arxiv.org/pdf/1912.01059)

## Usage

### Installation
```bash
git clone https://github.com/cgtuebingen/ggnn.git
cd ggnn
mkdir build
cd build
cmake ..
make -j4

```

### Example

You can copy ./src/ggnn_main_gpu_data.cu to the example folder in the source code for compilation.

```bash
make -j4
cd build

# Build Usage: 
./ggnn_main_gpu_data build base_file graph_dir degree tau 

#For example 
./ggnn_main_gpu_data build ../../../dataset/sift/base.fvecs ../../../graph/ggnn/sift 32 0.3 

# Search usage 
./ggnn_main_gpu_data query base_file query_file graph_dir degree top_k iteration measure topk_file 

#For example 
./ggnn_main_gpu_data query ../../..//dataset/sift/base.fvecs ../../../dataset/sift/query.fvecs ../../../graph/ggnn/sift 32 10 50 l2 ../../../result/ggnn/topk/sift/result.txt 

```
