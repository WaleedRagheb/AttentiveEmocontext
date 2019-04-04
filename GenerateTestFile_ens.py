import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pandas as pd
from AWDLSTM.create_toks_2 import *
from AWDEncClas.eval_clas import *
from AWDEncClas.bwd_ids_transformer import create_bw_data_test


#testfilePath = "data/root/dev.txt"
#classPath = "data/nlp_clas/train_ph1/train_ph1_clas"
#LMPath = "data/nlp_clas/train_ph1/train_ph1_lm"
#classModelName = 'ph1Class_recap_BEST'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'ph1LM2'
#targetTestOut = "data/nlp_clas/train_ph1/train_ph1_clas/tmp/tests/test.txt"


testfilePath = "data/root/dev.txt"
classPath = "data/nlp_clas/train_ALL/train_ALL_clas"
LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
classModelName = 'phClass_ALL'
#classModelName = 'ph1Class_ws_n_BEST_1'
LMModelName = 'phLM_ALL'
targetTestOut = "data/nlp_clas/train_ALL/train_ALL_clas/tmp/tests/test.txt"

#testfilePath = "data/root/dev.txt"
#classPath = "data/nlp_clas/train_ph2/train_ph2_clas"
#LMPath = "data/nlp_clas/train_ph2/train_ph2_lm"
#classModelName = 'ph2Class_ws_BEST'
#LMModelName = 'ph2LM'
#targetTestOut = "data/nlp_clas/train_ph2/train_ph2_clas/tmp/tests/test.txt"

#testfilePath = "data/root/dev.txt"
#classPath = "data/nlp_clas/train_phMerge/train_phMerge_class"
#LMPath = "data/nlp_clas/train_phMerge/train_phMerge_lm"
#classModelName = 'phMerClass_BEST'
#LMModelName = 'phMerLM'
#targetTestOut = "data/nlp_clas/train_phMerge/train_phMerge_class/tmp/tests/test.txt"


def f1_semE(preds, targs):
    #preds = torch.max(preds, dim=1)[1]
    precision, recall, fscore, support = precision_recall_fscore_support(targs, preds, average=None, labels=[0,1,2,3])
    #mrtrn = st.hmean([zip(*precision[1:],recall[1:])])
    #mrtrn = st.hmean([precision, recall])


    #  [ BOTH are The SAME]

    f1_filter = list(filter(lambda a: a != 0, fscore[1:]))
    mrtrn = st.hmean(f1_filter)

    #P_R = [*precision[1:], *recall[1:]]
    P_R = [sum(precision[1:])/3, sum(recall[1:])/3]
    mrtrn = st.hmean(P_R)

    return mrtrn


def generateTestOut(testfilePath, classPath, LMPath, classModelName, LMModelName, targetTestOut):
    label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
    emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

    targetDataFile_clas = classPath + "/tmp/tests/RealTest.csv"

    data = pd.read_csv(testfilePath, delimiter="\t", )
    idA,t1A,t2A,t3A,labA = [],[],[],[],[]
    with open(targetDataFile_clas, 'w', encoding="utf8") as f_clas:
        for id, lab, t1, t2, t3 in zip(data['id'],data['label'], data['turn1'], data['turn2'], data['turn3']):
            idA.append(id)
            t1A.append(t1)
            t2A.append(t2)
            t3A.append(t3)
            labA.append(emotion2label[lab])
            t1 = t1.replace("\"", "\"\"")
            t2 = t2.replace("\"", "\"\"")
            t3 = t3.replace("\"", "\"\"")

            f_clas.write(str(emotion2label[lab]))
            f_clas.write(f",\"{t1}\",\"{t2}\",\"{t3}\"\n")

            #f_clas.write(str(emotion2label[lab]) + ',')
            #f_clas.write("\"" + ' '.join([t1, t2, t3]) + "\"\n")

    df_val = pd.read_csv(targetDataFile_clas, header=None, chunksize=24000)
    tok_val, val_labels = get_all(df_val, 1)
    np.save(classPath + '/' + 'tmp/tests' + '/' + 'tok_val.npy', tok_val)
    np.save(classPath + '/' + 'tmp/tests' + '/' + 'lbl_val.npy', val_labels)
    tok_val = np.load(classPath + '/' + 'tmp/tests' + '/' + 'tok_val.npy')
    itos = pickle.load(open(LMPath + '/' + 'tmp' + '/' + 'itos.pkl', 'rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)
    val_clas = np.array([[stoi[o] for o in p] for p in tok_val])
    np.save(classPath + '/tmp/tests' + '/' + 'val_ids.npy', val_clas)

    create_bw_data_test(classPath + '/' + 'tmp/tests', LMPath + '/' + 'tmp' + '/itos.pkl' )

    prediction_fw, samplIdx_fw = eval_clas_eRisk(classPath, 0, lm_id=LMModelName, clas_id=classModelName, attention=True, bs=30, backwards=False, scores=True)
    prediction_bw, samplIdx_bw = eval_clas_eRisk(classPath, 0, lm_id=LMModelName, clas_id=classModelName,
                                                 attention=True, bs=30, backwards=True, scores=True)

    prediction_all =  [np.argmax(((softmax(p_fw))+(softmax(p_bw)))/2, axis=0) for p_fw, p_bw in zip(prediction_fw,prediction_bw)]

    idA = [idA[i] for i in samplIdx_fw]
    t1A = [t1A[i] for i in samplIdx_fw]
    t2A = [t2A[i] for i in samplIdx_fw]
    t3A = [t3A[i] for i in samplIdx_fw]
    labA = [labA[i] for i in samplIdx_fw]

    print("F1: " + str(f1_semE(labA, prediction_all)))

    with open(targetTestOut, 'w', encoding='utf8') as outF:
        outF.write('\t'.join(['id', 'turn1', 'turn2', 'turn3', 'label']) + "\n")
        for id,t1,t2,t3,labN in zip(idA, t1A, t2A, t3A, prediction_all):
            labT = label2emotion[labN]
            outF.write('\t'.join([str(id), t1, t2, t3, labT]) + "\n")






generateTestOut(testfilePath, classPath, LMPath, classModelName, LMModelName, targetTestOut)
