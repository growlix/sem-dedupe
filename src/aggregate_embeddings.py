import argparse
import os
import pickle as pkl
from typing import Tuple

import numpy as np

def aggregate_embeddings(data_dir, embedding_dim, delete_files=False, save_dir='', embedding_file_name='aggregated_embeddings.npy', uid_file_name='aggregated_ind2uid.pkl') -> Tuple[np.memmap, dict]:
    # Set save paths
    emb_save_path = os.path.join(save_dir, embedding_file_name)
    uid_save_path = os.path.join(save_dir, uid_file_name)
    # Data structures to collect embeddings and uids
    emb_arrays = []
    ind2uid = {}
    n_samples_per_file = []
    emb_array_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".npy")]
    uid2ind_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".pkl")]
    # Make sure there is a uid file for each embedding file
    if len(emb_array_files) != len(uid2ind_files):
        raise ValueError("Number of embedding files and uid files do not match")
    # Collect embeddings and uids
    for emb_file in emb_array_files:
        # Get common id for embedding and uid file
        file_id = os.path.splitext(os.path.basename(emb_file))[0].split("_")[1]
        uid_file = os.path.join(data_dir, f"ind2uid_{file_id}.pkl")
        if uid_file not in uid2ind_files:
            raise ValueError("No uid file for {}".format(emb_file))
        # Load embedding array
        emb_array = np.memmap(emb_file, dtype='float32', mode='r')
        # Reshape to 2D array
        emb_array = emb_array.reshape(-1, embedding_dim)
        print(f'Loaded {emb_file} with shape {emb_array.shape}')
        emb_arrays.append(emb_array)
        # Load uid
        with open(uid_file, 'rb') as f:
            uid_array = pkl.load(f)
        n_samples = emb_array.shape[0]
        # Ensure number of samples in embedding and uid file match
        if n_samples != len(uid_array):
            raise ValueError(f'Number of samples in {emb_file} and {uid_file} do not match')
        # Add offset to old indices and add to uid dict
        for old_ind, uid in uid_array.items():
            new_ind = old_ind + sum(n_samples_per_file)
            ind2uid[new_ind] = uid
        if delete_files:
            os.remove(uid_file)
            print(f'Deleted {uid_file}')
        n_samples_per_file.append(n_samples)
    with open(uid_save_path, 'wb') as f:
        pkl.dump(ind2uid, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(f'Saved uids to {uid_save_path}')
    dataset_len = sum(n_samples_per_file)
    # Destination memmap
    emb_array_aggregated = np.memmap(emb_save_path, dtype='float32', mode='w+', shape=(dataset_len, embedding_dim))
    # Iterate through all embedding arrays and add to destination memmap
    for i in range(len(emb_arrays)):
        emb_array = emb_arrays[i]
        offset = sum(n_samples_per_file[:i])
        n_samples = n_samples_per_file[i]
        emb_array_aggregated[offset:n_samples+offset, :] = emb_array
        emb_array_aggregated.flush()
        print(f'Added {n_samples} samples from {emb_array_files[i]} to {emb_save_path} at offset {offset}')
        print(emb_array_aggregated[offset,:])
        print(emb_array_aggregated[offset+n_samples-1,:])
        if delete_files:
            os.remove(emb_array_files[i])
            print(f'Deleted {emb_array_files[i]}')
    print(f'Saved aggregated embeddings to {emb_save_path} with shape {emb_array_aggregated.shape}')

    return emb_array_aggregated, ind2uid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate embeddings')
    parser.add_argument('--dir', type=str, help='Path to directory containing embeddings. Embeddings should be saved as embeddings_*.npy and uids as ind2uid_*.pkl, where * is identical for each pair of files')
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--delete_files', action='store_true', help='Delete files after aggregating')
    parser.add_argument('--save_dir', type=str, default='', help='Path to save aggregated embeddings')
    args = parser.parse_args()

    aggregate_embeddings(data_dir=args.dir, embedding_dim=args.embedding_dim, delete_files=args.delete_files, save_dir=args.save_dir)
