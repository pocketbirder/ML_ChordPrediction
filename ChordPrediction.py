import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

Chord2Code = {
    'Am' : -1,'A#m' : -2,'Bm' : -3,'Cm' : -4,'C#m' : -5,'Dm' : -6,'D#m' : -7,'Em' : -8,
    'Fm' : -9,'F#m' : -10,'Gm' : -11,'G#m' : -12,'None' : 0,'A' : 1,'Bb' : 2,'B' : 3,'C' : 4,
    'Db' : 5,'D' : 6,'Eb' : 7,'E' : 8,'F' : 9,'Gb' : 10,'G' : 11,'Ab' : 12}

Code2Chord = {v:k for k,v in Chord2Code.items()}


## Data read form txt

dataFile = open('./Dataset.txt','r')
rawdataString = dataFile.readlines()
rawdata = []

for line in rawdataString:
    dataItem = []
    word = ""
    for char in line:
        if(char == '\n'):
            dataItem.append(Chord2Code[word])
            word = ""
            rawdata.append(dataItem.copy())
            dataItem.clear()
        else:
            if (char == ' '):
                if(len(word)!=0):
                    dataItem.append(Chord2Code[word])
                word = ""
            else:
                word  += char

dataFile.close()

print(f'rawdata:{rawdata}')

## Data extend

X_dataset = []
Y_dataset = []
data = []
for dataItem in rawdata:
    for i in range(len(dataItem)):
        if(12-i > 0):
            for num in range((12-i-1)):
                xdata = []
                for k in range(24):
                    xdata.append(0)
                data.append(xdata.copy())
                xdata.clear()
        for j in range(i+1):
            k = 0
            while (k < 24):  ## 把一个和弦信息分解为24维的两个八度上的音 根音、三音、五音三个位为1，三音大调为mi，小调为降mi
                if(k == abs(dataItem[j])):
                    if(dataItem[j] > 0):
                        xdata.append(1)
                        xdata.append(0)
                        xdata.append(0)
                        xdata.append(0)
                        xdata.append(1)
                        xdata.append(0)
                        xdata.append(0)
                        xdata.append(1)
                    else:
                        xdata.append(1)
                        xdata.append(0)
                        xdata.append(0)
                        xdata.append(1)
                        xdata.append(0)
                        xdata.append(0)
                        xdata.append(0)
                        xdata.append(1)
                    k += 8
                else:
                    xdata.append(0)
                    k += 1
            data.append(xdata.copy())
            xdata.clear()
        X_dataset.append(data.copy())
        data.clear()
        y_data = []
        for j in range(25): ## Y集表示为 One-Hot 形式，对应正确和弦为1
            if(j == abs(dataItem[(i+1)%len(dataItem)]+12)):
                y_data.append(1)
            else:
                y_data.append(0)
        Y_dataset.append(y_data.copy())
        y_data.clear()

X_dataset = np.array(X_dataset)
Y_dataset = np.array(Y_dataset)

X_train_raw, X_val, Y_train_raw, Y_val = train_test_split(X_dataset,Y_dataset, test_size=0.2)##数据集分离为训练集与验证集

print(f'X_dataset:{X_dataset.shape}')
print(f'Y_dataset:{Y_dataset.shape}')

##转换数据集为TF张量
X_train = tf.convert_to_tensor(X_train_raw,dtype=tf.int32)
Y_train = tf.convert_to_tensor(Y_train_raw,dtype=tf.float32)
X_validation = tf.convert_to_tensor(X_val,dtype=tf.int32)
Y_validation = tf.convert_to_tensor(Y_val,dtype=tf.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],X_train.shape[2])),
    tf.keras.layers.LSTM(12, input_shape=X_train.shape[1:2]),
    tf.keras.layers.Dense(25,activation='sigmoid')
])

def Chord_On_Key_loss_function(y_true, y_pred):
    loss = tf.square(tf.multiply(y_true,y_pred))
    return tf.reduce_mean(loss,axis=-1)

def Chord_On_Key_accuracy(y_true, y_pred):
    pred = tf.argmax(y_pred)
    return

model.compile(optimizer='adam',
              loss='categorical_hinge',##hinge loss对应最大间隔算法
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100, verbose=2)##模型训练

ChordPredict = model.make_predict_function()

predictchord = X_validation
prediction = model.call(X_validation)
numRight = 0

for i in range(len(predictchord)):##测试在验证集上的正确性
    if(np.argmax(prediction[i]) == np.argmax(Y_dataset[i])):
        ChordSeq = []
        for j in predictchord[i]:
            if (np.argmax(j) != 0):##argmax返回01串第一个1的位置：根音
                MajOrMin = 1
                if(j[np.argmax(j)+3]):##判断和弦是否为小调
                    MajOrMin = -1
                ChordSeq.append(Code2Chord[MajOrMin * np.argmax(j)])
        print(f'Chord prediction of \"{ChordSeq}\" is \"{Code2Chord[np.argmax(prediction[i])-12]}\"')
        ChordSeq.clear()
        numRight += 1

print(f'Accuracy ={numRight/len(predictchord)}')


the_seq = [[[0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0]]]
ChordSeq = ['C','F']
while (len(the_seq[0]) < 12):
    the_seq_tf = tf.convert_to_tensor(the_seq,dtype=tf.int32)
    prediction = np.argmax(model.call(the_seq_tf)[0])
    ChordSeq.append(Code2Chord[prediction-12])
    ##print(f'Chord prediction of \"{ChordSeq}\" is \"{Code2Chord[np.argmax(prediction[i])-12]}\"')
    k = 0
    while (k < 24):  ## 把一个和弦信息分解为24维的两个八度上的音 根音、三音、五音三个位为1
        if(k == abs(prediction-12)):
            if(prediction > 0):
                xdata.append(1)
                xdata.append(0)
                xdata.append(0)
                xdata.append(0)
                xdata.append(1)
                xdata.append(0)
                xdata.append(0)
                xdata.append(1)
            else:
                xdata.append(1)
                xdata.append(0)
                xdata.append(0)
                xdata.append(1)
                xdata.append(0)
                xdata.append(0)
                xdata.append(0)
                xdata.append(1)
            k += 8
        else:
            xdata.append(0)
            k += 1
    the_seq[0].append(xdata.copy())
    xdata.clear()
print(ChordSeq)