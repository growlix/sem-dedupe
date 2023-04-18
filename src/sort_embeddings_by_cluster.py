# Distribute N clusters across S subsets: np.linspace(0, N, S) for E embeddings files.
# Each subset contains the embeddings for N/S clusters.
# Destination file structure looks like (if using 100 subsets):
# |--clustered_embeddings_unaggregated
# |  |--00
# |  |  |--clusters_000.pkl
# |  |  |--clusters_001.pkl
# |  |  | ...
# |  |  |--clusters_E.pkl
# |  |  |--clusterembeddings_000.npy
# |  |  |--clusterembeddings_001.npy
# |  |  | ...
# |  |  |--clusterembeddings_E.npy
# |  |  |--ind2uid_000.pkl
# |  |  |--ind2uid_001.pkl
# |  |  | ...
# |  |  |--ind2uid_E.pkl
# |  |  |--uid2ind_000.pkl
# |  |  |--uid2ind_001.pkl
# |  |  | ...
# |  |  |--uid2ind_E.pkl
# |  |--01
# |  |  |--clusters_000.pkl
# |  |  |--clusters_001.pkl
# |  |  | ...
# |  |  |--clusters_E.pkl
# |  |  |--clusterembeddings_000.npy
# |  |  |--clusterembeddings_001.npy
# |  |  | ...
# |  |  |--clusterembeddings_E.npy
# |  |  |--ind2uid_000.pkl
# |  |  |--ind2uid_001.pkl
# |  |  | ...
# |  |  |--ind2uid_E.pkl
# |  |  |--uid2ind_000.pkl
# |  |  |--uid2ind_001.pkl
# |  |  | ...
# |  |  |--uid2ind_E.pkl
# | ...
# |  |--99
# |  |  |--clusters_000.pkl
# |  |  |--clusters_001.pkl
# |  |  | ...
# |  |  |--clusters_E.pkl
# |  |  |--clusterembeddings_000.npy
# |  |  |--clusterembeddings_001.npy
# |  |  | ...
# |  |  |--clusterembeddings_E.npy
# |  |  |--ind2uid_000.pkl
# |  |  |--ind2uid_001.pkl
# |  |  | ...
# |  |  |--ind2uid_E.pkl
# |  |  |--uid2ind_000.pkl
# |  |  |--uid2ind_001.pkl
# |  |  | ...
# |  |  |--uid2ind_E.pkl

# After aggregating (see aggregate_cluster_embeddings.py), destination file organization looks like:
# |--clustered_embeddings
# |  |--00
# |  |  |--clusters.pkl
# |  |  |--clusterembeddings.npy
# |  |  |--ind2uid.pkl
# |  |  |--uid2ind.pkl
# |  |--01
# |  |  |--clusters.pkl
# |  |  |--clusterembeddings.npy
# |  |  |--ind2uid.pkl
# |  |  |--uid2ind.pkl
# | ...
# |  |--99
# |  |  |--clusters.pkl
# |  |  |--clusterembeddings.npy
# |  |  |--ind2uid.pkl
# |  |  |--uid2ind.pkl

import argparse
from functools import partial
import logging
import multiprocessing as mp
import os
import pickle as pkl
from typing import Union

import numpy as np
from tqdm import tqdm

from utils import load_and_infer_memmap

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def check_and_assign_sample(
        samples_to_check: dict, # Dict where samples_to_check['label'] is a list of samples in a cluster. We are going to check if they are in the embedding array
        uid2ind_emb: dict, # Mapping from sample uid to index in embedding array
        ):
    samples_to_check = samples_to_check['labels']
    # Indices of samples in the embedding array in the current cluster
    inds_in_cluster = []
    # Mapping from sample index in embedding array to sample uid
    ind2uid_emb = {}
    # Iterate through every sample in the cluster
    for sample_id in samples_to_check:
        # If sample is in the embedding array, update the data structures
        try:
            sample_ind = uid2ind_emb[sample_id]
            # Add embedding array index into set of indices for cluster
            inds_in_cluster.append(sample_ind)
            # Add sample index : uid mapping to ind2uid_emb
            ind2uid_emb[sample_ind] = sample_id
        # If sample is not in the embedding array, skip it
        except KeyError:
            pass
    return np.array(inds_in_cluster, dtype=int), ind2uid_emb


def sort_embeddings_by_cluster(emb_array: np.ndarray, cluster_ids: list, uid2ind_emb: dict, n_subsets: int=200, save_directory: str='cluster_embeddings', file_index: str='0'):

    # Create string representations of subset numbers
    subset_strings = [str(i).zfill(len(str(n_subsets))) for i in range(n_subsets)]
    # Assign clusters to subsets
    n_clusters = len(cluster_ids)
    subset_boundaries = np.linspace(0, n_clusters, n_subsets+1, dtype=int)
    subset_counts = np.diff(subset_boundaries)
    cluster2subset = np.arange(n_subsets).repeat(subset_counts) # Map from cluster to subset
    # Map from subset to list of clusters
    clusters_in_subset = {i: list(range(subset_boundaries[i], subset_boundaries[i+1])) for i in range(n_subsets)}
    # Mapping from index into embedding array to uid
    ind2uid_emb = {}

    # Create destination directories
    for i in range(n_subsets):
        subset_dir = os.path.join(save_directory, subset_strings[i])
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)

    # Store number of samples in each subset
    subset_sizes = {i: 0 for i in range(n_subsets)}
    # Store indices into embedding array for each cluster
    embedding_indices_for_each_cluster = {i : np.array([], dtype=int) for i in range(n_clusters)}
    # Parallelization seems slower due to passing around huge data structures
    # Partially apply uid2ind_emb to check_and_assign_sample
    # check_and_assign_sample_partial = partial(check_and_assign_sample, uid2ind_emb=uid2ind_emb)
    # with mp.Pool() as pool:
    #     assignment_results = pool.map(check_and_assign_sample_partial, tqdm(cluster_ids), chunksize=1000)
    # for i, cluster in tqdm(enumerate(assignment_results), total=len(cluster_ids)):
    #     # Add indices into list of embedding array indices for current cluster
    #     embedding_indices_for_each_cluster[i] = cluster[0]
    #     # Add sample index : uid mapping to ind2uid_emb
    #     ind2uid_emb.update(cluster[1])
    #     # Increment subset size
    #     subset_sizes[cluster2subset[i]] += len(cluster[0])
    # Iterate over clusters and get indices into embedding array of samples in each
    # cluster    
    for i, cluster in tqdm(enumerate(cluster_ids), total=len(cluster_ids)):
        inds_in_cluster, cluster_ind2uid_emb = check_and_assign_sample(cluster, uid2ind_emb)
        # Add indices into list of embedding array indices for current cluster
        embedding_indices_for_each_cluster[i] = inds_in_cluster
        # Add sample index : uid mapping to ind2uid_emb
        ind2uid_emb.update(cluster_ind2uid_emb)
        # Increment subset size
        subset_sizes[cluster2subset[i]] += len(inds_in_cluster)
    
    # Iterate over subsets and save embeddings for each cluster in subset
    for subset_i in tqdm(range(n_subsets)):
        subset_size = subset_sizes[subset_i]
        if subset_size > 1:
            subset_cluster_ids = []
            subset_dir = os.path.join(save_directory, subset_strings[subset_i])
            # Output file base
            emb_array_path_full = os.path.join(subset_dir,  f'clusterembeddings_{file_index}.npy')
            # Create array to store embeddings for subset
            subset_emb_array = np.memmap(emb_array_path_full, dtype='float32', mode='w+', shape=(subset_size, emb_array.shape[1]))
            # Store uid2ind for subset
            uid2ind_subset = {}
            # Store ind2uid for subset
            ind2uid_subset = {}
            # Store offset for indexing into subset embedding array
            offset = 0
            # Iterate over clusters in subset
            for cluster_i in clusters_in_subset[subset_i]:
                cluster_id = {'labels': [], 'inds': np.array([], dtype=int)}
                # Get indices into embedding array for current cluster
                cluster_indices = embedding_indices_for_each_cluster[cluster_i]
                # Get number of samples in cluster
                cluster_size = len(cluster_indices)
                # Add embeddings to subset embedding array
                subset_emb_array[offset:offset+cluster_size,:] = emb_array[cluster_indices, :]
                for sample_index_subset, sample_index_emb_array in enumerate(cluster_indices, offset):
                    # Get UID for sample
                    uid = ind2uid_emb[sample_index_emb_array]
                    # Update uid2ind_subset with sample index in subset embedding array
                    uid2ind_subset[uid] = sample_index_subset
                    # Update ind2uid_subset with uid
                    ind2uid_subset[sample_index_subset] = uid
                    # Update cluster ID with index and uid
                    cluster_id['inds'] = np.append(cluster_id['inds'], sample_index_subset)
                    cluster_id['labels'].append(uid)
                # Update subset cluster ID
                subset_cluster_ids.append(cluster_id)
                # Update offset
                offset += cluster_size
            # Flush memmap
            subset_emb_array.flush()
            logger.info(f'Wrote {subset_size} embeddings to {emb_array_path_full}')
            # Save uid2ind, ind2uid, and cluster data for subset
            for data, name in zip([subset_cluster_ids, uid2ind_subset, ind2uid_subset], ['clusters', 'uid2ind', 'ind2uid']):
                full_name = f'{name}_{file_index}.pkl'
                data_path = os.path.join(subset_dir, full_name)
                with open(data_path, 'wb') as f:
                    pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
                logger.info(f'Wrote {len(data)} {name} to {data_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sort embeddings into new files based on the cluster they belong to')
    parser.add_argument('--n_subsets', type=int, help='Number of subsets to split clusters into', default=200)
    parser.add_argument('--cluster_ids', type=str, help='Path to file containing sample ids for each cluster')
    parser.add_argument('--emb_path', type=str, help='Path to n x d numpy memmap where n = n samples and d = dimensionality')
    parser.add_argument('--n_samples', type=int, help='Number of samples in embedding array')
    parser.add_argument('--dim', type=int, help='Embedding dimensionality')
    parser.add_argument('--sample_mapping', type=str, default='index', help='Path to file mapping sample ids to indices in embedding array')
    # parser.add_argument('--invert_mapping', action='store_true', help='Invert the sample mapping (use if you have a mapping from indices to sample ids)')    
    parser.add_argument('--save_directory', type=str, default='clustered_embeddings_unaggregated')
    parser.add_argument('--file-index', type=str, help='Number to append to output file names. For example, if you are working on subset 060 of 099, set this to 060.')
    args = parser.parse_args()

    # Load embeddings
    emb_array = load_and_infer_memmap(data_path=args.emb_path, n_samples=args.n_samples, dim=args.dim)

    with open(args.sample_mapping, 'rb') as f:
        sample_mapping = pkl.load(f)
    if len(sample_mapping) != emb_array.shape[0]:
        raise ValueError(f"Number of sample ids ({len(sample_mapping)}) does not match number of samples ({args.n_samples})")
    logger.info(f'Loaded {len(sample_mapping)} sample ids from {args.sample_mapping}')

    with open(args.cluster_ids, 'rb') as f:
        cluster_ids = pkl.load(f)
    logger.info(f'Loaded {len(cluster_ids)} cluster ids from {args.cluster_ids}')
    
    sort_embeddings_by_cluster(emb_array=emb_array, cluster_ids=cluster_ids, uid2ind_emb=sample_mapping, n_subsets=args.n_subsets, save_directory=args.save_directory, file_index=args.file_index)