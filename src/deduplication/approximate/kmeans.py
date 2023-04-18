import argparse
import logging
import os
import pickle as pkl
import time
from typing import Union, List, Tuple

import faiss
import numpy as np
import torch
from torch.nn.functional import normalize
from tqdm import tqdm

from utils import load_and_infer_memmap

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

def kmeans_clustering(
    data: np.ndarray,
    sample_paths: Union[np.ndarray, dict],
    filename_base: str,
    save_directory: str='centroids',
    n_centroids: int=1000,
    niter: int=50,
    seed: int=1234,
    verbose: bool=True,
    max_points_per_centroid: int=256,
    save_kmeans: bool=True,
    centroids_file: Union[str, None]=None,
    index_file: Union[str, None]=None,
    train_only: bool=False,
    ) -> Union[Tuple[np.ndarray, List], None]:
    """
    Runs Kmeans clustering using 'faiss', and ranks each cluster/class items using the 
    distance to the cluster centorid. Saves the cluster data to disk.
    args:
        data (np.ndarray): Embeddings. Numpy array of shape [dataset_size x
            representation_dim]. Each embedding should be (row) normalized
        sample_paths Union[np.ndarray, dict]: Mapping from sample index to path. 
            sample_paths[i] is the path to (or label for) the ith sample.
        filename_base (str): 
        save_directory (str): directory into which centroid data should be saved
        n_centroids (int): number of centroids.
        niter (int): Kmeans clustering iterations.
        seed (int): Random seed
        max_points_per_centroid (int, Optional): Will not use more than this many data points per
            centroid when fitting.
        save_kmeans (bool, Optional): If True, saves the Kmeans object to disk. Default: True.
        centroids_file (str, Optional): If provided, loads the Kmeans centroids from disk.
        index_file (str, Optional): If provided, loads the Kmeans index from disk.
        train_only (bool, Optional): If True, only trains the Kmeans object, and does not
            cluster. Default: False.
    returns:
        nearest_cent: ndarray in which nearest_cent[i] corresponds to the cluster for the
            ith sample.
        centroid_data: List of length n_centroids, where centroid_data[i] is a dict in
            which centroid_data[i]['inds'] is the sample indices for the ith cluster, and
            centroid_data[i]['labels'] is the sample UIDs for the ith cluster.
            centroid_data is also saved to disk as .pkl file.
        
    """
    
    if (centroids_file is None) ^ (index_file is None):
        raise ValueError('Must provide both centroids_file and index_file, or neither')
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)    
    
    # Step 1) Compute Kmeans centroids
    d = data.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # spherical = True  # spherical=True when Kmeans_with_cos_dist is True
    # Instantiate Kmeans object
    kmeans = faiss.Kmeans(d, n_centroids, niter=niter, verbose=verbose, seed=seed, spherical=True, gpu=True, max_points_per_centroid=max_points_per_centroid)
    # Load Kmeans objects from disk
    if centroids_file is not None:
        with open(centroids_file, 'rb') as handle:
            centroids = np.load(handle)
        logger.info(f'Loaded Kmeans centroids from {centroids_file}')
        kmeans_index = faiss.read_index(index_file)
        logger.info(f'Loaded Kmeans index from {index_file}')
        if device == 'cuda':
            kmeans_index = faiss.index_cpu_to_all_gpus(kmeans_index)
            logger.info(f'Moved Kmeans index to {device}')
        kmeans.centroids = centroids
        kmeans.index = kmeans_index
        for i in range(n_centroids):
            if not all(kmeans.centroids[i, :] == kmeans.index.reconstruct(i)):
                raise ValueError('Centroid and index do not match. Check that the index and centroids are from the same Kmeans object.')
    # Train Kmeans
    else:
        logger.info(f'Clustering on {device} ....')
        st = time.time()
        kmeans.train(data)
        logger.info(f'Time for clustering (mins): {(time.time()-st)/(60)}')
        if save_kmeans:
            # Save centroids
            centroids_path = os.path.join(save_directory, f'{filename_base}_centroids.npy')
            with open(centroids_path, 'wb') as handle:
                np.save(handle, kmeans.centroids)
            logger.info(f'Saved Kmeans centroids to {centroids_path}')
            # Save index
            index_path = os.path.join(save_directory, f'{filename_base}.index')
            if device == 'cuda':
                save_index = faiss.index_gpu_to_cpu(kmeans.index)
            else:
                save_index = kmeans.index
            faiss.write_index(save_index, index_path)
            logger.info(f'Saved Kmeans index to {index_path}')

    if train_only:
        logger.info('Training only, not clustering')
        return None

    # Step 2) Find the nearest centroid for each data point, l2 distance search
    logger.info('Computing nearest centroids')
    st = time.time()
    dist_to_cent, nearest_cent = kmeans.index.search(data, 1) # nearest_cent: the nearest centroid for each example in data. dist_to_cent: contains the squared L2 distances.
    dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)
    logger.info(f'time to find nearest centroids: {(time.time()-st)/60}')

    logger.info('Grouping samples to centroids')
    centroid_data = []
    # Parallelize this
    for centroid_i in tqdm(range(len(kmeans.centroids))):
        centroid_inds = np.where(nearest_cent == centroid_i)[0]
        if isinstance(sample_paths, dict):
            centroid_paths = [sample_paths[ind] for ind in centroid_inds]
        else:
            centroid_paths = sample_paths[centroid_inds]
        centroid_data.append({"inds": centroid_inds, "labels": centroid_paths})
    savename = os.path.join(save_directory, f'{filename_base}_clusters.pkl')
    with open(savename, 'wb') as handle:
        pkl.dump(centroid_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info(f'Saved clustering data to {savename}')

    return nearest_cent, centroid_data
    # # Step 3) sort each class/cluster
    # logger.info('Ranking...')
    # st = time.time()
    # if use_clusters_bal: # for cluster balancing
    #     assert use_supervised_prototypes is False, 'use_clusters_bal requires use_supervised_prototypes=False'
    #     df = pd.DataFrame({'paths_list': sample_paths, 'nearest_cent':nearest_cent, 'dist_to_cent':dist_to_cent})
    #     sorted_clusters = rank_within_cluster_df(data, df, kmeans.centroids, sim_metric, keep_hard, spherical)
    #     # sorted_clusters = rank_within_cluster(data, paths_list, kmeans.centroids, nearest_cent, dist_to_cent, sim_metric, keep_hard, spherical)
    #     logger.info(f'time for ranking {(time.time()-st)/60} mins')
    #     return sorted_clusters


# TODO: Make this work (does not currently work)
def rank_within_cluster_df(data, df, centroids: np.ndarray, sim_metric: str, keep_hard: bool=True, spherical: bool=False) -> list:
    """
    Sorts each cluster items by the distance to the cluster centroid
    """

    assert sim_metric in ['cosine', 'l2'], 'sim_metric should be one of ["cosine", "l2"]'

    sorted_clusters = []
    for cluster_c in tqdm(range(len(centroids))): 
        # ids = (nearest_cent==cluster_c)  # boolean array: True for the examples in cluster c
        cluster_df = df.loc[df['nearest_cent'] == cluster_c]

        cluster_items = list(cluster_df.index) #np.where(ids)[0] # ids of examples in cluster c
        if sim_metric=='cosine':
            if spherical:
                cluster_dists_to_cent = list(1 - cluster_df['dist_to_cent'])
            else:
                cluster_c_centroid = torch.Tensor(centroids[cluster_c])
                sim_to_cent = torch.nn.CosineSimilarity(dim=1)(torch.Tensor(data[cluster_items]), cluster_c_centroid)
                cluster_dists_to_cent = (1-sim_to_cent).tolist()

        elif sim_metric=='l2': # get the l2 distance from "dist_to_cent" array
            cluster_dists_to_cent = list(cluster_df['dist_to_cent'])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        # in_labels = [img_name.split('_')[0] for img_name in paths_list[ids]]
        images_names = list(cluster_df['paths_list'])
        sort_descending = keep_hard
        cluster_sorted = sorted(zip(images_names, cluster_items, cluster_dists_to_cent, cluster_label), key=lambda x: x[2], reverse=sort_descending) # sort_descending = True for descending sort
            
        sorted_clusters.append(cluster_sorted) #  Descending dists. list of of list of tuples (example, dist). The ith list of tuples corresponds to cluster i
    
    return sorted_clusters


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='k-means on embeddings')
    parser.add_argument('--data_path', type=str, help='Path to n x d numpy memmap where n = n samples and d = dimensionality')
    parser.add_argument('--n_samples', type=int, help='Number of samples in embedding array')
    parser.add_argument('--sample_ids', type=str, default='index', help='Path to file containing index:uid mapping, or "index" to just use index in embedding array')
    parser.add_argument('--dim', type=int, help='Embedding dimensionality')
    parser.add_argument('--save_directory', type=str, default='/tmp/centroids/')
    parser.add_argument('--filename_base', type=str, default='')
    parser.add_argument('--n_centroids', type=int, default=50000, help='Number of centroids to use. See https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids for info on how to choose this.')
    parser.add_argument('--n_iter', type=int, default=40, help='Number of kmeans iterations')
    parser.add_argument('--max_points_per_centroid', type=int, default=256, help='Will not use more than this many data points per centroid when fitting.')
    parser.add_argument('--save_kmeans', action='store_false', help='Save kmeans objects to disk')
    parser.add_argument('--centroids_file', type=str, help='Path to saved kmeans centroids file')
    parser.add_argument('--index_file', type=str, help='Path to saved kmeans index file')
    parser.add_argument('--train_only', action='store_true', help='Only train kmeans, do not do clustering')
    args = parser.parse_args()

    # See the faiss wiki/faq for info on how to choose hparams for clustering: https://github.com/facebookresearch/faiss/wiki/

    emb_array = load_and_infer_memmap(args.data_path, args.n_samples, args.dim)
    
    if args.sample_ids == "index":
        sample_ids = np.arange(args.n_samples)
        logger.info(f'Using index as sample ids')
    else:
        with open(args.sample_ids, 'rb') as f:
            sample_ids = pkl.load(f)
        if len(sample_ids) != emb_array.shape[0]:
            raise ValueError(f"Number of sample ids ({len(sample_ids)}) does not match number of samples ({args.n_samples})")
        logger.info(f'Loaded {len(sample_ids)} sample ids from {args.sample_ids}')

    clusters = kmeans_clustering(
        emb_array,
        sample_ids,
        filename_base=args.filename_base,
        save_directory=args.save_directory,
        n_centroids=args.n_centroids,
        niter=args.n_iter,
        max_points_per_centroid=args.max_points_per_centroid,
        save_kmeans=args.save_kmeans,
        centroids_file=args.centroids_file,
        index_file=args.index_file,
        train_only=args.train_only
        )
