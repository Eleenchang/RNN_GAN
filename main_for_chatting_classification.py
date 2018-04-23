import RNN
import pandas as pd
import nltk, itertools, numpy
import Data_Preprocessor as DP

vocabulary_size = 500
filename = "chatting.csv"

f = open(filename, 'rb')

sentences = []
labels = []
for v in f:
    this = v.decode('cp949').split(",")
    labels = labels + [this[0]]
    sentences = sentences + [this[1]]

index_to_label = numpy.unique(labels)
label_to_index = {w: i for i, w in enumerate(index_to_label)}
y = [label_to_index[c] for c in labels]

preprocessor = DP.Data_Preprocessor("classification", dataset = sentences, vocabulary_size = vocabulary_size, minlen=5, label=y)
print(len(preprocessor.x_train))
print(len(preprocessor.y_train))

RNN = RNN.myRNN(nlayer=2, nclass=len(index_to_label))
RNN.training(preprocessor, maxit=10, batch_size=100)
pred = RNN.classification(preprocessor.x_train)
print(pred)
y = [y[i] for i in preprocessor.use_idx]
result = pd.DataFrame({"true" : y, "pred" : pred})
print(pd.crosstab(result.true, result.pred, margins=True))

