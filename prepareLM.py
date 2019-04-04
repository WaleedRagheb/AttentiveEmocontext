import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from AWDLSTM.create_toks_2 import *
from AWDLSTM.tok2id import *
from AWDEncClas.finetune_lm import*
from AWDEncClas.bwd_ids_transformer import create_bw_data

#create_toks("train_ph1","train_ph1_apr_cls", chunksize=24000, n_lbls=1)
#create_toks("train_phMerge","train_phMerge_apr_cls", chunksize=24000, n_lbls=1)
#create_toks("train_ALL","train_ALL_apr_cls", chunksize=24000, n_lbls=1)

#tok2id("train_ph1", min_freq=2)
#tok2id("train_phMerge", min_freq=2)
#tok2id("train_ALL", min_freq=2)

#create_bw_data('train_ph1')
#create_bw_data('train_ALL')

#train_lm('data/nlp_clas/train_ph1/train_ph1_lm', 'data/nlp_clas/wikitext-103_2', 0, lm_id='ph1LM2', backwards=False, cl=30)
#train_lm('data/nlp_clas/train_phMerge/train_phMerge_lm', 'data/nlp_clas/wikitext-103_2', 0, lm_id='phMerLM', backwards=False, cl=30)

#train_lm('data/nlp_clas/train_ph1/train_ph1_lm', 'data/nlp_clas/wikitext-103_2', 0, lm_id='ph1LM2', backwards=True, cl=30)

train_lm('data/nlp_clas/train_ALL/train_ALL_lm', 'data/nlp_clas/wikitext-103_2', 0, lm_id='phLM_ALL', backwards=False, cl=30)
#train_lm('data/nlp_clas/train_ALL/train_ALL_lm', 'data/nlp_clas/wikitext-103_2', 1, lm_id='phLM_ALL', backwards=True, cl=30)