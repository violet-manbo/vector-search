import numpy as np
import argparse
import os

def read_ivecs(filename):
    a = np.fromfile(filename, dtype='int32')
    dim = a[0]
    return a.reshape(-1, dim + 1)[:, 1:]

def read_retrieved_txt(filename, skip_head=3, skip_tail=0):
    with open(filename, 'r') as f:
        lines = f.readlines()
    valid_lines = lines[skip_head: len(lines) - skip_tail]
    return np.array([[int(x) for x in line.strip().split()] for line in valid_lines])

def compute_recall_at_k(gt, ret, k=10):
    total_recall = 0
    for g, r in zip(gt, ret):
        hit = len(np.intersect1d(g, r[:k]))
        total_recall += hit / len(g)
    return total_recall / len(gt)

def main():
    parser = argparse.ArgumentParser(description="Compute Recall@K and save to file")
    parser.add_argument('--gt', type=str, default='./sift_groundtruth.ivecs', help='Groundtruth .ivecs file path')
    parser.add_argument('--ret', type=str, required=True, help='Retrieved result file (e.g., output.log)')
    parser.add_argument('--out', type=str, required=True, help='Output file to write Recall result')
    parser.add_argument('--k', type=int, default=10, help='Top-K to compute recall at')
    args = parser.parse_args()

    groundtruth = read_ivecs(args.gt)[:, :args.k]
    retrieved = read_retrieved_txt(args.ret)

    assert groundtruth.shape[0] == retrieved.shape[0], \
        f"Groundtruth queries: {groundtruth.shape[0]}, Retrieved queries: {retrieved.shape[0]}"

    recall = compute_recall_at_k(groundtruth, retrieved, k=args.k)
    result_line = f"Recall@{args.k} = {recall:.4f}\n"

    print(result_line.strip())

    with open(args.out, 'w') as f:
        f.write(result_line)

if __name__ == '__main__':
    main()
