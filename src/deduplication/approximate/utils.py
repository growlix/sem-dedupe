import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

def load_and_infer_memmap(data_path: str, n_samples: Union[int, None]=None, dim: Union[int, None]=None, silent=False):
    # Load and attempt to infer n_samples or dim if only one is provided
    if n_samples is None and dim is None:
        raise ValueError("Must provide either n_samples or dim")
    if n_samples is None or dim is None:
        emb_array = np.memmap(data_path, dtype='float32', mode='r')
        if n_samples is None:
            assert dim is not None # Stupid type checking
            emb_array = emb_array.reshape(-1, dim)
        elif dim is None:
            emb_array = emb_array.reshape(n_samples, -1)
    else:
        emb_array = np.memmap(data_path, dtype='float32', mode='r', shape=(n_samples, dim))
    if not silent:
        logger.info(f'Loaded {emb_array.shape[0]} x {emb_array.shape[1]} embedding array from {data_path}')
    
    return emb_array