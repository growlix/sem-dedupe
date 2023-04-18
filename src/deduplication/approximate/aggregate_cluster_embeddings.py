# Distribute N clusters across S subsets: np.linspace(0, N, S) for E embeddings files.
# Each subset contains the embeddings for N/S clusters.
# Target file structure looks like (if using 100 subsets):
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

# After aggregating, destination file organization looks like:
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
import logging
import os
import pickle as pkl
from typing import Optional, Union

import numpy as np
from tqdm import tqdm

from utils import load_and_infer_memmap

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def sort_embeddings_by_cluster(
        data_directory: str,
        dim: int,
        emb_prefix: str='clusterembeddings',
        ind2uid_prefix: str='ind2uid',
        uid2ind_prefix: str='uid2ind',
        clusters_prefix: str='clusters',
        save_directory: str='clustered_embeddings',
        ):
    
    os.makedirs(save_directory, exist_ok=True)
    files = os.listdir(data_directory)
    file_info = {
        'emb': {'prefix': emb_prefix},
        'ind2uid': {'prefix': ind2uid_prefix, 'output': {}},
        'uid2ind': {'prefix': uid2ind_prefix, 'output': {}},
        'clusters': {'prefix': clusters_prefix, 'output': []}
    }
    n_files_per_type = ([],[])
    for file_type, file_type_info in file_info.items():
        prefix = file_type_info['prefix']
        files = sorted([f for f in os.listdir(data_directory) if prefix in f])
        file_info[file_type]['files'] = files
        n_files_per_type[0].append(prefix)
        n_files_per_type[1].append(len(files))
    
    # Check that there are the same number of files per type
    if not all([n == n_files_per_type[1][0] for n in n_files_per_type[1][1:]]):
        file_counts_string = '\n'.join([f'{prefix}: {n}' for prefix, n in zip(*n_files_per_type)])
        raise ValueError(f'Number of files per type does not match.\n{file_counts_string}')
    else:
        n_files_per_type = n_files_per_type[1][0]
        logger.info(f'Number of files per type: {n_files_per_type}')

    # Get number of embeddings
    n_embeddings = 0
    for file in file_info['emb']['files']:
        emb_array_small = load_and_infer_memmap(data_path=os.path.join(data_directory, file), n_samples=None, dim=dim, silent=True)
        n_embeddings += emb_array_small.shape[0]
    # Create memmap
    emb_array_big = np.memmap(filename=os.path.join(save_directory, file_info['emb']['prefix'] + '.npy'), shape=(n_embeddings, dim), dtype=np.float32, mode='w+')
    # Keep track of indexing offset into big array
    offset = 0

    # Iterate through files and aggregate
    for file_i in tqdm(range(n_files_per_type)):
        file_numbers = []
        n_embeddings_in_small = 0
        clusters_small = []
        ind2uid_small = {}
        uid2ind_small = {}
        for file_type, file_type_info in file_info.items():
            file = file_type_info['files'][file_i]
            file_numbers.append(file.split('_')[-1].split('.')[0])
            prefix = file_type_info['prefix']
            if file_type == 'emb':
                # Open small embedding array
                emb_array_small = load_and_infer_memmap(data_path=os.path.join(data_directory, file), n_samples=None, dim=dim, silent=True)
                # Number of embeddings in small array
                n_embeddings_in_small = emb_array_small.shape[0]
                # Copy small array into big array
                emb_array_big[offset:offset+n_embeddings_in_small,:] = emb_array_small
            if file_type == 'clusters':
                # Open clusters
                with open(os.path.join(data_directory, file), 'rb') as f:
                    clusters_small = pkl.load(f)
                if file_i == 0:
                    file_info['clusters']['output'] = clusters_small
                else:
                    clusters_big = file_info['clusters']['output']
                    # Iterate through each cluster
                    for cluster_big, cluster_small in zip(clusters_big, clusters_small):
                        # Add labels
                        cluster_big['labels'].extend(cluster_small['labels'])
                        # Add indices with offset
                        cluster_big['inds'] = np.concatenate([cluster_big['inds'], cluster_small['inds'] + offset], axis=0)
                    file_info['clusters']['output'] = clusters_big
            if file_type == 'ind2uid':
                # Open ind2uid
                with open(os.path.join(data_directory, file), 'rb') as f:
                    ind2uid_small = pkl.load(f)
                # Add to big ind2uid w/ updated offset
                file_info['ind2uid']['output'].update({k + offset: v for k, v in ind2uid_small.items()})
            if file_type == 'uid2ind':
                # Open uid2ind
                with open(os.path.join(data_directory, file), 'rb') as f:
                    uid2ind_small = pkl.load(f)
                # Add to big uid2ind w/ updated offset
                file_info['uid2ind']['output'].update({k : v + offset for k, v in uid2ind_small.items()})
        # Sanity checks
        # Check all file numbers are the same
        if not all([file_numbers[0] == file_number for file_number in file_numbers[1:]]):
            raise ValueError(f'File numbers do not match: {file_numbers}')
        # Check that uids and inds are the same for all three files
        for cluster in clusters_small:
            for ind, uid in zip(cluster['inds'], cluster['labels']):
                if ind2uid_small[ind] != uid:
                    raise ValueError(f'Index and UID do not match for ind {ind} and uid {uid} across ind2uid, uid2ind, and cluster files for file number {file_numbers[0]}')
                if uid2ind_small[uid] != ind:
                    raise ValueError(f'Index and UID do not match for ind {ind} and uid {uid} across ind2uid, uid2ind, and cluster files for file number {file_numbers[0]}')
        # Increment offset
        offset += n_embeddings_in_small
    
    # Flush array to disk
    logger.info(f'Flushing aggregated embeddings to disk')
    emb_array_big.flush()
    logger.info(f'Wrote aggregated embeddings to {os.path.join(save_directory, file_info["emb"]["prefix"] + ".npy")}')
    # Write output files
    for file_type, file_type_info in file_info.items():
        if file_type != 'emb':
            save_path = os.path.join(save_directory, file_type_info['prefix'] + '.pkl')
            with open(save_path, 'wb') as f:
                pkl.dump(file_type_info['output'], f)
            logger.info(f'Wrote {file_type} file to {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate a directory of clustered embeddings and associated files')
    parser.add_argument('--data_directory', type=str, help='Directory containing clustered embeddings and associated files')
    parser.add_argument('--emb_prefix', type=str, help='Prefix for n x d numpy memmap file where n = n samples and d = dimensionality', default='clusterembeddings')
    parser.add_argument('--dim', type=int, help='Embedding dimensionality')
    parser.add_argument('--ind2uid_prefix', type=str, default='ind2uid', help='Prefix for file mapping indices in embedding array to sample ids')
    parser.add_argument('--uid2ind_prefix', type=str, default='uid2ind', help='Prefix for file mapping sample ids to indices in embedding array')
    parser.add_argument('--clusters_prefix', type=str, default='clusters', help='Prefix for file containing sample ids for each cluster')
    parser.add_argument('--save_directory', type=str, default='clustered_embeddings', help='Directory in which to save aggregated files')
    args = parser.parse_args()

    # TODO: Return value
    sort_embeddings_by_cluster(**vars(args))