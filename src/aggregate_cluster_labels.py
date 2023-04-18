import argparse
from functools import partial
from multiprocessing.pool import Pool
import os
import pickle as pkl
from typing import List, Tuple

from tqdm import tqdm


def aggregate_clusters(data_dir, delete_files=False, save_dir='', save_name='aggregated_clusters'):
    aggregated_clusters = []
    n_clusters = None
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".pkl")]

    # TODO: Parallelize. Collect stats.
    for file in tqdm(files):
        with open(file, 'rb') as f:
            clusters= pkl.load(f)
        print(f'Loaded {file}')
        # Find n clusters
        if n_clusters is None:
            n_clusters = len(clusters)
            # Initialize aggregated clusters
            for _ in range(n_clusters):
                aggregated_clusters.append({'labels': []})
        # Raise error if n clusters does not match
        elif n_clusters != len(clusters):
            raise ValueError(f'Number of clusters in {file} does not match previous files')
        # Add labels to aggregated clusters
        for i, cluster in enumerate(clusters):
            aggregated_clusters[i]['labels'].extend(cluster['labels'])
        if delete_files:
            os.remove(file)
            print(f'Deleted {file}')
    # Write files
    save_path_full = os.path.join(save_dir, save_name + '.pkl')
    with open(save_path_full, 'wb') as f:
        pkl.dump(aggregated_clusters, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(f'Saved aggregated clusters to {save_path_full}')
    return aggregated_clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate clusters')
    parser.add_argument('--data_dir', type=str, help='Path to directory containing cluster files, which must end with .pkl')
    parser.add_argument('--delete_files', action='store_true', help='Delete files after aggregating')
    parser.add_argument('--save_path', type=str, default='', help='Path to save aggregated clusters')
    parser.add_argument('--save_name', type=str, default='aggregated_clusters', help='Name of aggregated clusters file. Will have .pkl appended.')
    args = parser.parse_args()

    aggregate_clusters(data_dir=args.data_dir, delete_files=args.delete_files, save_dir=args.save_path, save_name=args.save_name)