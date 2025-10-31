# SONG

## Overview 

SONG is a graph-based approximate nearest neighbor search algorithm. It implements the graph searching algorithm and optimizations of [SONG: Approximate Nearest Neighbor Search on GPU](http://research.baidu.com/Public/uploads/5f5c37aa9c37c.pdf)

## Usage

### Installation
``` bash

git clone git@github.com:sunbelbd/song.git
cd song
./build_template.sh

```

### Fill Parameters
```
Usage: ./fill_parameters.sh <pq_size> <dim> <cos/l2/ip>
For example: ./fill_parameters.sh 100 32 l2
```

`<dim>` is the number of dimensions of the dataset.

`<pq_size>` is a searching parameter. The greater it is, the better recall we obtain in the searching result (with the cost of longer running time).

`<cos/l2/ip>` is the similarity/distance measure. `cos` for Cosine similarity, `l2` for L2 distance/Euclidean distance, and `ip` for max inner-product search.

### TEST

After completing the previous step, You can run the following script to generate the executable file.

```
# Build Usage: ./build_graph.sh <build_data> <row> <dimension> <l2/ip/cos>
For example: ./build_graph.sh ../dataset/sift/sift_base.libsvm 1000000 128 l2

# Search Usage: ./test_query.sh <query_data> <built_graph_row> <built_graph_dimension> <l2/ip/cos> [display top k]
For example: ./test_query.sh ../dataset/sift/sift_query.libsvm 1000000 32 l2 5
```
The similarity/distance measure should match the parameter in `fill_parameters.sh`.
