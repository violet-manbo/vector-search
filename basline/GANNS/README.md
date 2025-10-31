# GANNS

## Overview 
GANNS is a GPU-based algorithm which can accelerate the ANN search on proximity graphs by re-designing the classical CPU-based search algorithm. It is based on the method proposed in the paper(https://opus.lib.uts.edu.au/bitstream/10453/166490/3/GPU-accelerated%20Proximity%20Graph%20Approximate%20Nearest%20Neighbor%20Search%20and%20Construction.pdf)

## Usage

### Installation
```bash
git clone git@github.com:yuyuanhang/GANNS.git
cd GANNS
```

### Example
Step 1. Generate template
```bash
./generate_template.sh
```

Step2. Build Graph
To use construction algorithm, generate build instance.
```bash
./generate_build_instances.sh [dim] [metric]
./build_128_l2 [base_path] [graph_type] [e] [d_min]
```
For example: 
```bash
./generate_build_instace.sh 128 l2
./build_128_l2 ../dataset/sift/base.fvecs nsw 60 16
```

Step3. Search 
The same to build:
```bash
./generate_query_instances.sh [dim] [metric]
./query_128_l2 [base_path] [query_path] [graph_type] [graph_path] [groundtruth_path] [e] [k]
```
for example


