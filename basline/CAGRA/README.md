# SONG

## Overview 

GGNN performs nearest-neighbor computations on CUDA-capable GPUs. It is based on the method proposed in the paper [GGNN: Graph-based GPU Nearest Neighbor Search](https://arxiv.org/pdf/1912.01059)

The algorithm is based on the method proposed in the paper[Cagra: Highly parallel graph construction and approximate nearest neighbor search for gpus](https://arxiv.org/pdf/2308.15136)

## Usage

### Installation
```bash
git clone git@github.com:rapidsai/cuvs.git
cd cuvs
conda env create --name cuvs -f conda/environments/all_cuda-128_arch-x86_64.yaml
conda activate cuvs
./build.sh libcuvs --no-mg

```

### Example

You can copy ./src/my_cagra.cu  into the cuvs/example/cpp in cuvs for compilation.

```bash
cd example/cpp
mkdir build
cd build
make -j4

# Build Usage: 
./my_cagra build base_file graph degree

#For example 
./my_cagra build ./dataset/sift/base.fvecs ./graph/cagra/sift 32

# Search usage 
./my_cagra search index_file query_file.fvecs iterations topk_file search_width internal_topk_size team_size

#For example 
./my_cagra search ./graph/cagra/sift ./dataset/sift/query.fvecs 10 ./result/sift/topk 1 64 8 

```
