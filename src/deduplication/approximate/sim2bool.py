import argparse
import os
import logging
import pickle as pkl
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert similarity file to boolean keep/exclude based on threshold(s).')
parser.add_argument('--data_file', type=str)
parser.add_argument('--save_dir', default='.', type=str)
parser.add_argument('--keep', nargs='+', help='One or more similarity thresholds below which samples are kept. Writes one file per threshold.', type=float)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig()

# Load file
with open(args.data_file, 'rb') as handle:
    similarities = pkl.load(handle)
logger.info(f'Loaded {len(similarities["similarities"])} values from {args.data_file}')

# Ensure that desired quantiles are present in similarities file
for keep_fraction in args.keep:
    if keep_fraction not in similarities['quantiles'].keys():
        raise KeyError(f'{keep_fraction} quantile not present in {args.data_file}. Please use one of:\n{similarities["quantiles"].keys()}')

# Iterate through keep thresholds
for keep_fraction in args.keep:
    logger.info(f'Creating keep file for {keep_fraction}...')
    threshold = similarities['quantiles'][keep_fraction]
    keep_dict = {
        'quantiles': similarities['quantiles'],
        'similarities': {}
    }
    i = 0
    for uid, similarity in tqdm(similarities['similarities'].items()):
        if similarity < threshold:
            keep_dict['similarities'][uid] = True
        else:
            keep_dict['similarities'][uid] = False
        if i % 10000000 == 0:
            logger.info(f'Processed {i} samples...')
        i += 1
    filename = os.path.join(args.save_dir, f'keep_{str(keep_fraction).replace(".","pt")}.pkl')
    logger.info(f'Writing keep file to {filename}...')
    # Make save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    with open(filename, 'wb') as handle:
        pkl.dump(keep_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info(f'Wrote {len(keep_dict["similarities"])} values to {filename}')
