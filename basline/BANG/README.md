# BANG
## Overview 
This project has three different strategies for datasets of varying sizes. In our experiment, we test the base algorithm BANG-BASE. The details of the algorithm are shown in paper[BANG: Billion-Scale Approximate NearestNeighbour Search using a Single GPU](https://arxiv.org/pdf/2401.11324)

## Installation

### Build DiskANN and Graph Constraction
This algorithm uses DiskANN to build a VAMANA graph. So we need to clone DiskANN firstly
```bash
git clone git@github.com:microsoft/DiskANN.git
cd DiskANN
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
sudo apt install libmkl-full-dev
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 

```
Next, we can use it to build the base graph.
```bash
./apps/build_disk_index --data_type float --dist_fn l2 --data_path ./dataset/sift/sift_base.fvecs --index_path_prefix ./graph/sift -R 32 -L 50 -B 10 -M 10
```
### Installation for BANG
```bash
git clone git@github.com:karthik86248/BANG-Billion-Scale-ANN.git
cd BANG-Billion-Scale-ANN/BANG_BASE
mkdir build 
cd build
make -j4
```

### Data Preprocess
BANG needs to process the VAMANA graph from DiskANN before the search phase.
```bash

# usage ./bang_preprocess.py index_by_diskann index_for_bang dim data_type(0->uint8, 1->int8, 2->float) R,  for example, we use the graph built above:
./bang_preprocecss.py ./graph/sift_disk.index ./graph/sift_disk.bin 128 2 32

```
### Search 
```bash
#usage ./bang_search index_prefix query.bin groundtruth.bin n_queries topk data_type metric for example:
./bang_search ./graph/sift/sift ./dataset/sift/base.bin ./dataset/sift/gt.bin float l2

```

