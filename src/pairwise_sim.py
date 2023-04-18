import argparse
from functools import partial
import logging
import os
import pickle as pkl
from typing import Mapping, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from utils import load_and_infer_memmap

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

def compute_similarities(
    cluster: Mapping,
    embedding_path: str,
    n_samples: Union[int, None]=None,
    dim: Union[int, None]=None,
    reduction: str="max", # TODO: Allow more than just max
    sample_ids: Union[dict, None]=None,
    ):
    embeddings = load_and_infer_memmap(data_path=embedding_path, n_samples=None, dim=dim, silent=True)
    centroid_inds = cluster['inds']
    centroid_embeddings = torch.tensor(embeddings[centroid_inds,:])
    centroid_size = centroid_embeddings.shape[0]
    similarity_vector = torch.tensor([])
    labels = []
    if centroid_size > 1:
        similarity_matrix = (centroid_embeddings @ (centroid_embeddings.T))
        similarity_matrix.fill_diagonal_(0.0)
        assert similarity_matrix.shape[0]==similarity_matrix.shape[1]
        triu = torch.triu(similarity_matrix, diagonal=1)
        # TODO: Choose between mean or max
        similarity_vector = triu.max(dim=0)[0]
        if sample_ids is not None:
            labels = []
            for ind in centroid_inds:
                labels.append(sample_ids[ind])
        else:
            labels = cluster['labels']
    elif centroid_size == 1:
        similarity_vector = torch.tensor([0.0])
        labels = cluster['labels']
    return similarity_vector.numpy(), labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Distances between each pair of samples in each cluster.')
    parser.add_argument('--emb_path', type=str, help='Path to embeddings file, a n x d numpy memmap where n = n samples and d = dimensionality')
    parser.add_argument('--cluster_data', '--centroid_path', type=str, help='Path to cluster data')
    parser.add_argument('--sample_ids', type=str, default='', help='Path to file containing index:label mapping; no value will use index in embedding array')
    parser.add_argument('--n_samples', type=int, help='Number of samples in embedding array')
    parser.add_argument('--dim', type=int, help='Embedding dimensionality')
    parser.add_argument('--save_directory', type=str, default='/tmp/similarities/')
    parser.add_argument('--filename_base', type=str, default='')
    parser.add_argument('--quantiles', nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], help="Quantiles to compute")
    parser.add_argument('--subsample', type=int, default=-1, help="Number of clusters to subsample. Useful if you want a faster distribution estimate and don't need to compute distances for all samples.")
    args = parser.parse_args()
    
    # This program computes the cosine simlarity between each pair of samples in each
    # cluster. It then takes the max similarity value for each sample and assigns that to
    # be the sample's similarity value. It then computes the quantiles of the similarity
    # values for all samples (more specifically, in computes max(triu(similarity_matrix,
    # diagonal=1), dim=0) for each cluster). It saves a dictionary of the form
    # {"quantiles": quantiles, "similarities": similarities}, in which "quantiles" is
    # mapping where keys = quantile values (computed over 0.05 : 1 : 0.05) and values =
    # similarity value for that quantile, and "similarities" is a mapping from sample id
    # to similarity value.

    # The Pile
    # train
    # n_samples = 210607728
    # val
    # args.n_samples = 214670

    # C4
    # train
    # n_samples = 364868892
    # val
    # n_samples = 364608

    # args.emb_path = "/tmp/pile_val_embeddings.npy"
    # args.centroid_path = "/tmp/pile_val_clusters.pkl"
    # args.n_samples = 210607728
    # args.dim = 768

    # emb_array = np.memmap(args.emb_path, dtype='float32', mode='r', shape=(args.n_samples, args.dim))
    with open(args.cluster_data, 'rb') as handle:
        clusters = pkl.load(handle)
    logger.info(f'Loaded {len(clusters)} clusters from {args.cluster_data}')
    
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)    

    # Load IDs
    sample_ids = None
    if args.sample_ids:
        with open(args.sample_ids, 'rb') as handle:
            sample_ids = pkl.load(handle)
        logger.info(f'Loaded {len(sample_ids)} sample ids from {args.sample_ids}')
    else:
        logger.info(f'Using index as sample id')

    # Confirm number of indices and uids match in cluster file
    n_cluster_ids = 0
    n_cluster_inds = 0
    for cluster in clusters:
        n_cluster_ids += len(cluster['labels'])
        n_cluster_inds += len(cluster['inds'])
    if n_cluster_ids != n_cluster_inds:
        raise ValueError(f'Number of cluster labels ({n_cluster_ids}) does not match number of cluster indices ({n_cluster_inds}) in {args.cluster_data}')
    # Confirm number matches between cluster and index file
    if sample_ids is not None:
        if n_cluster_ids != len(sample_ids):
            raise ValueError(f'Number of cluster labels ({n_cluster_ids}) does not match number of cluster indices ({len(sample_ids)}) in {args.cluster_data} and {args.sample_ids}')

    if args.subsample == -1:
        clusters_to_sample = len(clusters)
    else:
        clusters_to_sample = args.subsample
    quantiles = args.quantiles

    all_similarities = []
    similarities_map = {}
    i = 0
    mapfun = partial(compute_similarities, embedding_path=args.emb_path, n_samples=args.n_samples, dim=args.dim, sample_ids=sample_ids)
    with mp.Pool() as pool:
        for cluster_similarity, cluster_labels in tqdm(pool.imap_unordered(mapfun, clusters[:clusters_to_sample]), total=clusters_to_sample):
            all_similarities.append(cluster_similarity)
            # TODO: Check for collisions
            if len(cluster_labels) != len(cluster_similarity):
                raise ValueError(f'Number of labels ({len(cluster_labels)}) does not match number of similarities ({len(cluster_similarity)})')
            for label, similarity in zip(cluster_labels, cluster_similarity):
                similarities_map[label] = similarity
            # wtf is this
            if cluster_similarity.size == 0:
                print('Clu')
            logger.info(f'Completed {i} clusters')
            i += 1
    logger.info(f'Finished computing similarities for {len(similarities_map)} samples. Computing quantiles...')
    all_similarities = np.concatenate(all_similarities, axis=0)
    similarity_quantiles = np.quantile(all_similarities, quantiles)
    quantile_map = {}
    for computed, q in zip(similarity_quantiles, quantiles):
        quantile_map[q] = computed
    save_file = {
        "quantiles": quantile_map,
        "similarities": similarities_map
    }
    savename = os.path.join(args.save_directory, f'{args.filename_base}_max_similarity.pkl')
    logger.info(f'Quantiles computed. Saving similarity data to {savename}')
    with open(savename, 'wb') as handle:
        pkl.dump(save_file, handle, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info(f'Saved similarity data to {savename}')
