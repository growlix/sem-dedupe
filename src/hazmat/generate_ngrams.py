# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Generates all n-grams in a dataset. Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/scripts/clean_training_data/generate_13_grams.py

Loops through all documents and uses the logic found in janitor.py to extract n-grams.
We bucket each n-gram by hash into separate file buckets to allow easy parallel processing in the
next stage. We also include the current pile document_id with each ngram instance to allow the
filtering to exclude n-grams that match more then 10 unique documents (done further down the pipeline).

Arguments
---------
--working_directory (-dir)
    Directory containing the pile distribution. An "output" subdirectory will be created underneath
    to store the bucketed 13-grams, checkpoint and done files. Default: current directory
--n_value (-n)
    n value in n-gram, added for later use if ever needed. Default: 13
--bucket_count (-buckets)
    Number of file buckets to use when generating n-grams. Default: 500
"""

import argparse
import json
import pickle
import os
import sys
from pathlib import Path
import glob
import signal
from signal import SIGINT
from typing import Any, Dict, Tuple

from tqdm import tqdm

from janitor import Janitor, word_ngrams
from archiver import TextArchive

import logging
from tqdm_multiprocess.logger import setup_logger_tqdm

from streaming import StreamingDataLoader, StreamingDataset

from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

terminate = False


def handler(signal_received, frame):
    global terminate
    terminate = True


# Hash buckets > disk backed files. Supports file position checkpointing and resuming
# Allows you to write continuously and checkpoint intermittently. If a failure occurs
# the buckets are simply truncated at your last checkpoint.
class Buckets:
    def __init__(self, directory, num_buckets):
        self.bucket_files = [
            os.path.join(directory, f"ngrams_{i}.bkt.txt") for i in range(num_buckets)
        ]
        self.buckets = list(map(TextArchive, self.bucket_files))
        self.checkpoint_file = os.path.join(directory, f"bucket_offsets.ckpt")

        if os.path.exists(self.checkpoint_file):
            self.bucket_offsets = pickle.load(open(self.checkpoint_file, "rb"))
        else:
            self.bucket_offsets = [0 for i in range(len(self.buckets))]

        for i, offset in enumerate(self.bucket_offsets):
            bucket = self.buckets[i]
            bucket.fh.seek(offset)
            bucket.fh.truncate()

    def add_data(self, key, value):
        i = hash(key) % len(self.buckets)
        bucket = self.buckets[i]
        bucket.add_data(value)

    def save_checkpoint(self):
        for bucket in self.buckets:
            bucket.fh.flush()

        bucket_offsets = [bucket.fh.tell() for bucket in self.buckets]
        pickle.dump(bucket_offsets, open(self.checkpoint_file, "wb"))

    def close_buckets(self):
        for bucket in self.buckets:
            bucket.commit()

class StreamingDatasetIndexed(StreamingDataset):
    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], int]:
        """Get sample by global index.
        Args:
            index (int): Sample index.
        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard], index

def do_ngrams_in_buckets(
    dataset: StreamingDataset,
    working_directory: str,
    checkpoint_name: str = 'ngram_count',
    n_value: int = 13,
    bucket_count: int = 500) -> None:

    def collate_fn(batch):
        return batch
    dataloader = StreamingDataLoader(dataset, batch_size=768, num_workers=8, collate_fn=collate_fn, shuffle=False)

    output_directory = os.path.join(working_directory, 'output', f'rank{rank}')
    os.makedirs(output_directory, exist_ok=True)

    logger.info(f'Generating {n_value}-grams and bucketing.')

    # Done file
    done_file = os.path.join(output_directory, f'ngram_buckets_rank{rank}.done')
    if os.path.exists(done_file):
        logger.info('ngrams already generated and bucketed, skipping')
        return

    # Checkpoint
    checkpoint_file = os.path.join(working_directory, 'checkpoints', f'{checkpoint_name}_rank{rank}.ckpt')
    if os.path.exists(checkpoint_file):
        checkpoint_offset = pickle.load(open(checkpoint_file, 'rb'))
    else:
        checkpoint_offset = 0

    logger.info(f'Starting at sample index {checkpoint_offset}')
    buckets = Buckets(output_directory, bucket_count)

    janitor = Janitor()
    batch_size = 1000
    batch_counter = 0

    dataloader_size = len(dataset)
    remaining = dataloader_size - checkpoint_offset
    index = 0
    for sample in tqdm(dataset, total=remaining):
        # batch = dataloader[i]
        # for sample, index in batch:
        text = sample[0]['text']
        # Save checkpoint every "batch_size", only allow terminate after checkpoint
        if batch_counter == batch_size:
            batch_counter = 0
            buckets.save_checkpoint()
            pickle.dump(index, open(checkpoint_file, 'wb'))
            if terminate:
                buckets.close_buckets()
                return

        ngrams = word_ngrams(janitor.normalize_string(text), n_value)
        for ngram in ngrams:
            buckets.add_data(ngram, f'{ngram} {index}')

        batch_counter += 1
        index += 1

    buckets.close_buckets()
    Path(done_file).touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 13 grams from a dataset.")
    parser.add_argument("-dir", "--working_directory", default="")
    parser.add_argument("-n", "--n_value", type=int, default=13)
    parser.add_argument("-buckets", "--bucket_count", type=int, default=500)
    parser.add_argument('--streaming_remote', type=str, default="None")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--split', type=str, default="train")

    if "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "0":
        print("Setting PYTHONHASHSEED=0")
        os.environ["PYTHONHASHSEED"] = "0"

    # Handle sigint (ctrl-c) cleanly
    previous_signal_int = signal.signal(SIGINT, handler)

    logfile_path = "ngrams.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()

    dataset = StreamingDatasetIndexed(
        local = args.streaming_local,
        remote = args.streaming_remote,
        split = args.split,
        shuffle = False
    )

    do_ngrams_in_buckets(
        dataset = dataset,
        n_value = args.n_value,
        working_directory = args.working_directory,
        bucket_count = args.bucket_count)

    info_dict = {"title": "dataset ngrams", "ngram_size": 13}
    info_dict_path = os.path.join(args.working_directory, "info.json")
    json.dump(info_dict, open(info_dict_path, "w"))
