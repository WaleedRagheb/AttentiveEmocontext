import pandas as pd

trainFilePath = "data/root/train.txt"
testFilePath = "data/root/dev.txt"
classPath = "data/nlp_clas/train_ph2/train_ph2_clas"
lmPath = "data/nlp_clas/train_ph2/train_ph2_lm"

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}



def prepareFiles(sourceDataFile, targetDataFile_clas, targetDataFile_lm, em2labDic = emotion2label):
    data = pd.read_csv(sourceDataFile, delimiter ="\t", )
    with open(targetDataFile_clas, 'w', encoding="utf8") as f_clas, open(targetDataFile_lm, 'w', encoding="utf8") as f_lm:
        for lab, t1, t2, t3 in zip(data['label'], data['turn1'],data['turn2'],data['turn3']):
            t1 = t1.replace("\"","\"\"")
            t2 = t2.replace("\"", "\"\"")
            t3 = t3.replace("\"", "\"\"")

            #f_clas.write(str(emotion2label[lab]))
            #f_lm.write("0")
            #f_clas.write(f",\"{t1}\",\"{t2}\",\"{t3}\"\n")
            #f_lm.write(f",\"{t1}\",\"{t2}\",\"{t3}\"\n")


            f_clas.write(str(emotion2label[lab]) + ',')
            f_lm.write("0,")
            f_clas.write("\"" + ' '.join([t1,t2,t3]) + "\"\n")
            f_lm.write("\"" + ' '.join([t1,t2,t3]) + "\"\n")



prepareFiles(trainFilePath, classPath + "/train.csv", lmPath + "/train.csv")
prepareFiles(testFilePath, classPath + "/test.csv", lmPath + "/test.csv")
