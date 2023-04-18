Fast clustering requires FAISS to be [installed from source w/ C++
dependencies](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md). You can
avoid this by using the
[growlix/data-dedupe](https://hub.docker.com/r/growlix/data-dedupe) docker image, which
has all the FAISS dependencies pre-installed.

### If your dataset + embeddings fit on disk on a single machine/node
0. Convert your dataset to [MosaicML's
   StreamingDataset](https://github.com/mosaicml/streaming) format and `docker pull growlix/data-dedupe`.

1. Run `get_embeddings.py`. This generates embeddings for each sample in the dataset and
   saves them to a [numpy
   memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).
   
   _How it works:_
   
   The embeddings are obtained using [E5 base](https://huggingface.co/intfloat/e5-base).
   The text of a data sample is expected to be contained in the `'text'` key of the data
   sample. The sample text is tokenized and broken into chunks of 512 tokens (the maximum
   sequence length of the embedding model), embeddings are obtained for each chunk, and
   chunks from the same sample are averaged togther to yield a single embedding for each
   sample. A separate process is launched for each available gpu, and the results are
   merged at the end. Mappings from index into dataset : sample uid and sample uid : index
   into dataset are also saved. If samples don't have UIDs, then the index into the
   dataset _is_ the UID (this is inadvisable).
   
   _Things to be mindful of:_
   
   Long samples may result in many many chunks. In order to avoid OOMing, there are
   separate batch sizes for the dataloader and inference.
   
   `StreamingDataset` is deterministic so _in theory_ it's ok to use sample indices as
   UIDs (IFF you are running everything on a single node), but I advise against this as a
   matter of hygiene.

   If you want to use a different embedding model, you will have to modify the code to
   change which model is instantiated (e5 is currently hard-coded and this is gross and
   I'm sorry), and add a collator and tokenizer to the respective registries.

2. Run `kmeans.py`. This fits centroids to your embeddings and/or assigns embeddings to
   your fitted centroids. If you are fitting centroids and assigning embeddings, it will
   save the sample indices and sample UIDs for each cluster. If you are only fitting
   centroids, it will save the centroids and the [FAISS
   index](https://github.com/facebookresearch/faiss/wiki/Getting-started#building-an-index-and-adding-the-vectors-to-it),
   both of which are needed to assign embeddings to clusters.

   See the [faiss
   wiki/faq](https://github.com/facebookresearch/faiss/wiki/) for info on FAISS and how to choose
   hparams for clustering. 

3. Run `pairwise_sim.py`. This computes the cosine similarity for each pair of samples within
   a cluster. It then takes the max similarity value for each sample and assigns that to
   be the sample's similarity value (specifically, it computes
   `max(triu(similarity_matrix, diagonal=1), dim=0)` for each cluster.). It then computes the
   quantiles of the similarity values for all samples. It saves a dictionary of the form
   `{"quantiles": {quantile: similarity}, "similarities": {uid: similarity}}`, in which
   `quantiles` is a mapping where keys are quantile values (computed over 0.05 : 1 : 0.05)
   and values are the similarity value for that quantile (e.g. `quantiles[0.75]` is the
   similarity value that corresponds to the 75th percentile of the similarity value
   distribution), and `similarities` is a mapping from sample uid to similarity value.

4. Run `write_and_prune_dataset.py`. This reads samples from a `StreamingDataset`, checks
   the sample's UID in the sample_uid : similarity mapping produced by `pairwise_sim.py`, and
   writes samples to a new `StreamingDataset` if the sample's similarity value is less
   than a given percentile threshold.

5. Train on your pruned dataset and enjoy the pareto improvement!

### If your dataset + embeddings don't fit on disk and/or you want to parallelize this process
Coming soon