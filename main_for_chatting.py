import RNN
import nltk, itertools, numpy

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
this_person_sens = [sentences[i] for i in range(len(labels)) if labels[i] == index_to_label[2]]
sentences = this_person_sens

RNN = RNN.myRNN(nlayer=2)
RNN.training(sentences, vocabulary_size = 600)
RNN.word_generate(nsens=10)