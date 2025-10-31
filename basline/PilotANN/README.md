# PilotANN

## Overview 
This algorithm is a memory-bounded GPU acceleration for graph-based approximate nearest neighbor search. You can check the original paper on https://arxiv.org/pdf/2503.21206. The source code is published on https://github.com/ytgui/PilotANN


## Installations and Usage

### Build Faiss
``` bash
# Download source
git clone https://github.com/facebookresearch/faiss/archive/refs/tags/v1.8.0.tar.gz
tar -xvf v1.8.0.tar.gz && cd faiss-*

# Build faiss
mkdir build && cd build
cmake .. -DFAISS_OPT_LEVEL='avx2' \
         -DFAISS_ENABLE_GPU=OFF   \
         -DFAISS_ENABLE_PYTHON=ON \
         -DCMAKE_BUILD_TYPE=Release

# Install
cd faiss/python && python3 ./setup.py install
```

### Build PilotANN
``` bash
# Download source
git clone git@github.com:ytgui/PilotANN.git
cd PilotANN
python ./setup.py develop
```


### EXAMPLE
You can run the following command to test if the installation was successful.
``` bash
python ./bench_1.py --checkpoint sift.ckpt --algorithm nsg --method search-pilot --d_principle 128 --n_neighbors 32 --top_k 10 --sample_ratio 0.25

```
