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
parser.add_argument('--keep_fraction', type=float, help='Threshold for similarity. Only similarities below threshold are kept and a set() of kept UIDs is saved.')
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

keep_fraction = args.keep_fraction

# Find all .pkl files in all subdirectories
pkl_files = []
for root, dirs, files in os.walk(args.data_dir):
    for file in files:
        if file.endswith(".pkl"):
             pkl_files.append(os.path.join(root, file))
pkl_files = sorted(pkl_files)
logger.info(f'Found {len(pkl_files)} .pkl files')

# Iterate through files first time to deteremine quantiles
quantiles = {}
elapsed_samples = 0
for pkl_file in tqdm(pkl_files):
    curr_samples = 0
    with open(pkl_file, 'rb') as handle:
        curr_sim = pkl.load(handle)
    logger.info(f'Loaded {pkl_file} with {len(curr_sim["similarities"])} similarities')
    if keep_fraction not in curr_sim['quantiles'].keys():
        raise KeyError(f'{keep_fraction} quantile not present in {args.data_file}. Please use one of:\n{curr_sim["quantiles"].keys()}')
    # If it's the first time
    curr_samples = len(curr_sim['similarities'])
    if not quantiles:
        quantiles = curr_sim['quantiles']
        
    # Otherwise, we need to merge things
    else:
        curr_quantiles = curr_sim['quantiles']
        # Compute running average of quantiles
        for q, curr_v in curr_quantiles.items():
            quantiles[q] = (quantiles[q]*elapsed_samples + curr_v*curr_samples) / (elapsed_samples + curr_samples)
        # Merge similarities
    elapsed_samples += curr_samples

# Iterate through files and merge
similarities = {'quantiles': quantiles, 'similarities': set()}
threshold = quantiles[keep_fraction]
logger.info(f'Threshold for keep fraction of {keep_fraction}: {threshold}')
kept = 0
removed = 0
for pkl_file in tqdm(pkl_files):
    curr_samples = 0
    with open(pkl_file, 'rb') as handle:
        curr_sim = pkl.load(handle)
    logger.info(f'Loaded {pkl_file} with {len(curr_sim["similarities"])} similarities')
    curr_samples = len(curr_sim['similarities'])
    # Update similarities
    for uid, sim in curr_sim['similarities'].items():
        if sim < threshold:
            similarities['similarities'].add(uid)
            kept += 1
        else:
            removed += 1
    if args.delete_after_merge:
        os.remove(pkl_file)
        logger.info(f'Deleted {pkl_file}')
    del curr_sim
similarities['kept'] = kept
similarities['removed'] = removed
logger.info(f'Kept {kept} similarities and removed {removed} similarities (kept {kept/(kept+removed)} of samples.')

logger.info(f'Writing similarities...')
# Make save directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)
file_name = os.path.join(args.save_dir, f'similarities_{str(keep_fraction).replace(".","pt")}.pkl')
with open(file_name, 'wb') as handle:
    pkl.dump(similarities, handle, protocol=pkl.HIGHEST_PROTOCOL)
logger.info(f'Wrote {len(similarities["similarities"])} similarities to {file_name}')