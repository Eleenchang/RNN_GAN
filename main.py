import RNN
import nltk, itertools
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

vocabulary_size = 200
#filename = "kimth.csv"
filename = "horror.csv"

f = open(filename, 'rb')
sentences = []
for v in f:
    #print(v)
    #print(v.decode('cp949'))
    #sentences = sentences + ["%s %s %s" % (sentence_start_token, v.decode('cp949').encode('utf-8'), sentence_end_token)]
    sentences = sentences + ["%s %s %s" % (sentence_start_token, v.decode('cp949'), sentence_end_token)]

tokenized_sens = [sens.split(" ") for sens in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sens))
vocab = word_freq.most_common(vocabulary_size - 1)
# vocab = word_freq.most_common()
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
voca_size = len(index_to_word)
print("voca_size = %d" % (voca_size))
word_to_index = {w: i for i, w in enumerate(index_to_word)}
for i, sent in enumerate(tokenized_sens):
    tokenized_sens[i] = [w if w in index_to_word else unknown_token for w in sent]

maxlen = 30
minlen = 5
temp_sens = []
for i in range(len(tokenized_sens)):
    if len(tokenized_sens[i]) <= maxlen and len(tokenized_sens[i]) >= minlen:
        temp_sens = temp_sens + [tokenized_sens[i]]

tokenized_sens = temp_sens

maxlen = 0
for i in range(len(tokenized_sens)):
    if maxlen < len(tokenized_sens[i]):
        maxlen = len(tokenized_sens[i])

maxlen = 70
print("maxlen = %d" % (maxlen))
print("no_of_sens = %d" % (len(tokenized_sens)))

x_train = [[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sens]
y_train = [[word_to_index[w] for w in sent[1:]] for sent in tokenized_sens]
print([index_to_word[i] for i in y_train[0]])
print([index_to_word[i] for i in y_train[1]])
print([index_to_word[i] for i in y_train[2]])

RNN = RNN.myRNN(voca_size, maxlen, word_to_index, index_to_word)
RNN.training(x_train, y_train, 100, "generator")
RNN.word_generate(nsens=10)