import argparse
import os
import logging
import pickle as pkl
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Aggregate similarity files')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_dir', default='.', type=str)
parser.add_argument('--delete_after_merge', action='store_true')
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

# Find all .pkl files in all subdirectories
pkl_files = []
for root, dirs, files in os.walk(args.data_dir):
    for file in files:
        if file.endswith(".pkl"):
             pkl_files.append(os.path.join(root, file))
pkl_files = sorted(pkl_files)
logger.info(f'Found {len(pkl_files)} .pkl files')

# Iterate through files and merge
similarities = {}
elapsed_samples = 0
for pkl_file in tqdm(pkl_files):
    curr_samples = 0
    with open(pkl_file, 'rb') as handle:
        curr_sim = pkl.load(handle)
    logger.info(f'Loaded {pkl_file} with {len(curr_sim["similarities"])} similarities')
    # Not much to do if it's the first file
    if not similarities:
        similarities = curr_sim
        curr_samples = len(curr_sim['similarities'])
    # Otherwise, we need to merge things
    else:
        curr_samples = len(curr_sim['similarities'])
        curr_quantiles = curr_sim['quantiles']
        # Compute running average of quantiles
        for q, curr_v in curr_quantiles.items():
            similarities['quantiles'][q] = (similarities['quantiles'][q]*elapsed_samples + curr_v*curr_samples) / (elapsed_samples + curr_samples)
        # Merge similarities
        similarities['similarities'].update(curr_sim['similarities'])
    elapsed_samples += curr_samples
    del curr_sim
    if args.delete_after_merge:
        os.remove(pkl_file)
        logger.info(f'Deleted {pkl_file}')


logger.info(f'Writing similarities...')
# Make save directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, 'similarities.pkl'), 'wb') as handle:
    pkl.dump(similarities, handle, protocol=pkl.HIGHEST_PROTOCOL)
logger.info(f'Wrote {len(similarities["similarities"])} similarities to {os.path.join(args.save_dir, "similarities.pkl")}')