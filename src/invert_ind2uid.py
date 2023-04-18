import os
import pickle as pkl
import sys

data_subset_string = sys.argv[1]

get_command = 'oci os object get -bn mosaicml-internal-dataset-mc4 --file ind2uid.pkl --name sem-dedupe/23-03-30/embeddings/ind2uid_***.pkl'
put_command = 'oci os object put -bn mosaicml-internal-dataset-mc4 --file uid2ind.pkl --name sem-dedupe/23-03-30/embeddings/uid2ind_***.pkl'
del_command = 'rm ind2uid.pkl uid2ind.pkl'

os.system(get_command.replace('***', data_subset_string))
with open('ind2uid.pkl', 'rb') as f:
    ind2uid = pkl.load(f)
uid2ind = {uid: ind for ind, uid in ind2uid.items()}
with open('uid2ind.pkl', 'wb') as f:
    pkl.dump(uid2ind, f, protocol=pkl.HIGHEST_PROTOCOL)
os.system(put_command.replace('***', data_subset_string))
os.system(del_command.replace('***', data_subset_string))