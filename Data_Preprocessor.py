import nltk, itertools, numpy

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

class Data_Preprocessor :

    def __init__(self, method, dataset, vocabulary_size, maxlen = None, minlen = 1, label = None):

        self.method = method
        self.dummy = vocabulary_size + 1
        self.vocabulary_size = vocabulary_size
        sentences = []

        for v in dataset:
            sentences = sentences + ["%s %s %s" % (sentence_start_token, v, sentence_end_token)]

        tokenized_sens, use_idx = self.tokenizing(sentences, vocabulary_size, maxlen, minlen)
        self.use_idx = use_idx


        if self.method == "generator" :
            x_train = [[self.word_to_index[w] for w in sent[:-1]] for sent in tokenized_sens]
            y_train = [[self.word_to_index[w] for w in sent[1:]] for sent in tokenized_sens]
            self.x_train = x_train
            self.y_train = y_train

        else :
            x_train = [[self.word_to_index[w] for w in sent] for sent in tokenized_sens]
            label = [label[i] for i in use_idx]
            self.x_train = x_train
            self.y_train = label

    def tokenizing(self, sentences, vocabulary_size, maxlen, minlen):

        tokenized_sens = [sens.split(" ") for sens in sentences]

        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sens))
        print(word_freq)

        if vocabulary_size == None :
            vocab = word_freq.most_common()
        else:
            vocab = word_freq.most_common(vocabulary_size-1)

        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        voca_size = len(index_to_word)
        print("voca_size = %d" % (voca_size))
        word_to_index = {w: i for i, w in enumerate(index_to_word)}
        for i, sent in enumerate(tokenized_sens):
            tokenized_sens[i] = [w if w in index_to_word else unknown_token for w in sent]

        if maxlen is None :
            maxlen = 0
            for i in range(len(tokenized_sens)):
                if maxlen < len(tokenized_sens[i]):
                    maxlen = len(tokenized_sens[i])

        temp_sens = []
        indices = []
        for i in range(len(tokenized_sens)):
            if len(tokenized_sens[i]) <= maxlen and len(tokenized_sens[i]) >= minlen:
                temp_sens = temp_sens + [tokenized_sens[i]]
                indices = indices + [i]

        tokenized_sens = temp_sens

        print("maxlen = %d" % (maxlen))
        print("no_of_sens = %d" % (len(tokenized_sens)))

        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.voca_size = voca_size
        self.max_len = maxlen
        self.min_len = minlen

        return(tokenized_sens, indices)

    def dummy_generation(self):

        lens = []
        for i in range(len(self.x_train)) :
            lens = lens + [len(self.x_train[i])]
            if len(self.x_train[i]) < self.max_len :
                self.x_train[i] = self.x_train[i]+ [self.dummy] * (self.max_len - len(self.x_train[i]))

        self.len_sens = lens