#! /usr/bin/env python3
import sys
import logging
import faiss

import torch
import numpy as np

# modified from https://github.com/facebookresearch/deepcluster
def get_knn(
    reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False
):
    """
    Finds the k elements in reference_embeddings that are closest to each
    element of test_embeddings.
    Args:
        reference_embeddings: numpy array of size (num_samples, dimensionality).
        test_embeddings: numpy array of size (num_samples2, dimensionality).
        k: int, number of nearest neighbors to find
        embeddings_come_from_same_source: if True, then the nearest neighbor of
                                         each element (which is actually itself)
                                         will be ignored.
    """
    d = reference_embeddings.shape[1]
    logging.info("running k-nn with k=%d"%k)
    logging.info("embedding dimensionality is %d"%d)
    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)
    _, indices = index.search(test_embeddings, k + 1)
    if embeddings_come_from_same_source:
        return indices[:, 1:]
    return indices[:, :k]

xb = np.random.random((6000, 256)).astype('float32')  # 数据库向量
b = get_knn(xb, xb, 1000, True)
print(b.shape)