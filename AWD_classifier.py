import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from AWDEncClas.train_clas import*



#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_ws_n', use_att=True, cl=100, use_clr=True, useWeightSampler = True, dropmult=1, bs=128)

#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_ws_1', use_att=True, cl=100, use_clr=True, useWeightSampler = False, dropmult=1, bs=128)
#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_ws_2', use_att=True, cl=100, use_clr=True, useWeightSampler = True, dropmult=0.9, bs=128)
#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_ws_3', use_att=True, cl=100, use_clr=True, useWeightSampler = True, dropmult=0.8, bs=128)
#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_ws_4', use_att=True, cl=100, use_clr=True, useWeightSampler = True, dropmult=1.1, bs=128)
#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_ws_5', use_att=True, cl=100, use_clr=True, useWeightSampler = True, dropmult=1.2, bs=128)


#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_Mask', use_att=True, cl=100, use_clr=True, useWeightSampler = True, dropmult=1, bs=128)
#train_clas('data/nlp_clas/train_ph2/train_ph2_clas', 1, lm_id='ph2LM', clas_id='ph2Class_ws', use_att=False, cl=50, use_clr=True, useWeightSampler = True)

#train_clas('data/nlp_clas/train_noFT/train_noFT_clas', 0, lm_id='wt103', clas_id='ph_noFT', use_att=True, cl=100, use_clr=False, useWeightSampler = True, dropmult=1, bs=128)

#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM2', clas_id='ph1Class_ws_6', use_att=True, cl=100, use_clr=True, useWeightSampler = True, dropmult=1, bs=128)

#train_clas('data/nlp_clas/train_phMerge/train_phMerge_class', 0, lm_id='phMerLM', clas_id='phMerClass', use_att=False, cl=100, use_clr=True, useWeightSampler = False, dropmult=1, bs=128)

#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM', clas_id='ph1Class_ws_n', use_att=True, cl=50, use_clr=True, useWeightSampler = True, dropmult=1, bs=128)

#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM2', clas_id='ph1Class_recap', use_att=True, cl=50, use_clr=True, useWeightSampler = True, dropmult=1, bs=128)
#train_clas('data/nlp_clas/train_ph1/train_ph1_clas', 0, lm_id='ph1LM2', clas_id='ph1Class_recap', use_att=True, cl=50, use_clr=True, useWeightSampler = True, dropmult=1, bs=128,backwards=True)

#train_clas('data/nlp_clas/train_ALL/train_ALL_clas', 0, lm_id='phLM_ALL', clas_id='phClass_ALL', use_att=True, cl=50, use_clr=True, useWeightSampler = True, dropmult=1, bs=128)
#train_clas('data/nlp_clas/train_ALL/train_ALL_clas', 1, lm_id='phLM_ALL', clas_id='phClass_ALL', use_att=True, cl=50, use_clr=True, useWeightSampler = True, dropmult=1, bs=128, backwards=True)