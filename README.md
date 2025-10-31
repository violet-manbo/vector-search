# A Benchmark For GPU-based ANN search
## Overview
This repository conducts unified testing and comparison of various GPU-based graph structure approximate nearest neighbor (ANN) search algorithms. The aim is to evaluate the performance, recall rate, and GPU resource usage of different algorithms in diferent datasets.

The covered algorithms include: SONG,CAGRA, GANNS, BANG, GGNN, PilotANN, etc.

## Repository Structure
```
├── algorithms/          # Installation of algorithms
│   ├── song/
│   ├── cagra/
│   ├── GANNS/
│   ├── BANG/
│   ├── ggnn/
│   └── pilotann/
├── configs/             # Experimental parameter configuration file
├── scripts/             # Running scripts
├── result/              # Results of all algorithms              
└── README.md
 ```

## Environment
* CUDA version >= 11.8
* gcc and g++ 11.0 or higher (C++11 support)
* Boost C++ libraries version >=1.74
* Python 3.10
* cmake (>= 3.25.2)

## Datasets
SIFT and GIST datasets can be downloaded from http://corpus-texmex.irisa.fr/

GLOVE200 and NYTIMES can be downloaded from https://github.com/erikbern/ann-benchmarks/blob/master/README.md

MNIST8M can be downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist8m

DEEP100M is to be cut out from DEEP1B. Take first 100M points. https://big-ann-benchmarks.com/


## Algorithms
The specific construction methods for each algorithm can be found in the README.md file in the corresponding directory.

## Run Experiments

### Installation

We present how to install each algorithm in the README.md under the folder baseline. 

### Running Experiments

Different algorithms have different build and execution pipelines:

| Algorithm | Build Type | Run Command |
|------------|-------------|--------------|
| **SONG** | Precompiled executable | `bash scripts/run_song.sh` |
| **GANNS** | Precompiled executable | `bash scripts/run_cagra.sh` |
| **GGNN** | Library + wrapper | `bash scripts/run_ggnn.sh` |
| **CAGRA** | Library + wrapper | `bash scripts/run_cagra.sh` |
| **BANG** | Requires compilation | `bash scripts/run_bang.sh` |
| **PilotANN** | Python scripts | `bash scripts/run_pilot.sh` |

### Build Parameters
All the construction parameters for the algorithms are listed in the folder config.

### Search
We launch the experiments by using the scripts in the folder Scrpits.
For example:
```bash
./Scripts/run_ggnn.sh
```

### Acknowledgement & Citation
This repository relies on the following open-source implementations:
* SONG
* GGNN
* GANNS
* cuVS
* BANG
* PilotANN