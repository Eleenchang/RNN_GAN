import tensorflow as tf
import numpy
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


class myRNN:
    def __init__(self, nlayer, nclass = None):

        self.nlayer = nlayer
        self.nclass = nclass

        if self.nclass == None :
            self.method = "generator"
        else :
            self.method = "classifier"

    def data_description(self, x_train, y_train = None):
        if self.method == "generator" :
            return(self.data_description_for_generator(x_train, y_train))
        else :
            return(self.data_description_for_classification(x_train, y_train))

    def data_description_for_generator(self, x_train, y_train):
        mask = []
        for i in range(len(x_train)):
            mask = mask + [1] * len(x_train[i])
            while len(x_train[i]) < self.max_len:
                x_train[i] = x_train[i] + [self.dummy]
                y_train[i] = y_train[i] + [self.dummy]
                mask = mask + [0]

        return [x_train, y_train, mask]

    def data_description_for_classification(self, x_train, y_train):

        y_lens = []
        mask = []

        for i in range(len(x_train)):
            mask = mask + [0] * (len(x_train[i])-1)
            mask = mask + [1]
            # this_mask = [0] * self.max_len
            # this_mask[-1] = 1
            # mask = mask + this_mask
            if y_train != None :
                y_lens = y_lens + [[y_train[i]]*len(x_train[i])]

            while len(x_train[i]) < self.max_len:
                x_train[i] = x_train[i] + [self.dummy]
                if y_train != None :
                    y_lens[i] = y_lens[i] + [y_train[i]]
                mask = mask + [0]

        return [x_train, y_lens, mask]

    def training(self, preprocessor, vocabulary_size = None, batch_size = 100, p_dropout = 0.1, maxit = 100):

        self.max_len = preprocessor.max_len
        self.min_len = preprocessor.min_len

        self.index_to_word = preprocessor.index_to_word
        self.word_to_index = preprocessor.word_to_index
        self.voca_size = preprocessor.voca_size

        if self.method == "generator" :
            num_units = self.voca_size
        else :
            num_units = self.nclass

        cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units), output_keep_prob=1 - p_dropout)]
        if self.nlayer > 1:
            for i in range(self.nlayer - 2):
                cell.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units),
                                                          output_keep_prob=1 - p_dropout))

        self.RNN_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cell)
        self.dummy = self.voca_size + 1

        self.x_train = preprocessor.x_train
        self.y_train = preprocessor.y_train

        if self.method == "generator" :
            return self.training_for_generator_model(batch_size, maxit)
        else :
            return self.training_for_classification_model(batch_size, maxit)


    def training_for_classification_model(self, batch_size, maxit):

        self.x_ = tf.placeholder(dtype=tf.int64, shape=[None, self.max_len])
        self.y_ = tf.placeholder(dtype=tf.int64, shape=[None, self.max_len])
        self.mask_ = tf.placeholder(dtype=tf.float32, shape=[None, ])

        self.prediction = self.RNN_initialize_and_prediction_fun(batch_size)

        y_one_hot = tf.one_hot(self.y_, depth=self.nclass)
        self.loss = self.cost(y_one_hot, self.prediction, self.mask_)

        self.optimization(maxit)


    def RNN_initialize_and_prediction_fun(self, batch_size, out = "prob", just_zero = False):
        self.batch_size = batch_size

        x_one_hot = tf.one_hot(self.x_, depth=self.voca_size)
        init_state = self.RNN_cell.zero_state(batch_size, tf.float32)
        if just_zero :
            return

        output, state = tf.nn.dynamic_rnn(cell=self.RNN_cell, inputs=x_one_hot, initial_state=init_state,
                                          dtype=tf.float32)
        if out == "prob" :
            return(tf.nn.softmax(logits=output))
        else :
            return(tf.argmax(tf.nn.softmax(logits=output), 2))

    def training_for_generator_model(self, batch_size, maxit):

        self.x_ = tf.placeholder(dtype=tf.int64, shape=[None, self.max_len])
        self.y_ = tf.placeholder(dtype=tf.int64, shape=[None, self.max_len])
        self.mask_ = tf.placeholder(dtype=tf.float32, shape=[None, ])

        self.prediction = self.RNN_initialize_and_prediction_fun(batch_size)

        y_one_hot = tf.one_hot(self.y_, depth=self.voca_size)
        self.loss = self.cost(y_one_hot, self.prediction, self.mask_)

        self.optimization(maxit)


    def optimization(self, maxit, n_epoch = 200, eps = 0.0001):
        train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(self.loss)

        self.sess = tf.Session()
        writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
        writer.add_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.initialize_all_variables())

        epoch = 0
        prev_cost = 1000
        flag_out = False
        for i in range(maxit):
            start_ind = 0
            end_ind = start_ind + self.batch_size
            j = 0
            while end_ind < len(self.x_train):

                this_batch_x, this_batch_y, this_batch_mask = self.data_description(self.x_train[start_ind:end_ind],
                                                                                    self.y_train[start_ind:end_ind])
                # print(len(this_batch_x), len(this_batch_y), len(this_batch_mask))
                feed = {self.x_: this_batch_x, self.y_: this_batch_y, self.mask_: this_batch_mask}
                self.sess.run(train_op, feed_dict=feed)
                current_cost = self.sess.run(self.loss, feed_dict=feed)
                start_ind = end_ind
                end_ind = end_ind + self.batch_size
                if abs(prev_cost - current_cost) / current_cost < eps:
                    epoch = epoch + 1

                prev_cost = current_cost

                if j % 10 == 0:
                    print(current_cost)

                j = j + 1

                if epoch > n_epoch:
                    flag_out = True

                if flag_out:
                    break

            if flag_out:
                break

    def cost(self, y, yhat, mask):
        cross_entrophy = y * tf.log(yhat)
        cross_entrophy = -tf.reduce_sum(cross_entrophy, reduction_indices=2)
        cross_entrophy = tf.reshape(tf.concat(axis = 1, values = cross_entrophy), [-1, ])
        cross_entrophy = tf.reduce_sum(cross_entrophy * mask)
        cross_entrophy = cross_entrophy / tf.reduce_sum(mask)

        return cross_entrophy

    def word_generate(self, nsens):

        self.prediction = self.RNN_initialize_and_prediction_fun(nsens)

        init_value = [sentence_start_token] * self.max_len
        init_value = [self.word_to_index[c] for c in init_value]
        init_value = [init_value] * self.batch_size

        init_value = numpy.asarray(init_value)
        print(init_value.shape)
        additional = numpy.zeros([self.batch_size, 1], dtype="int64")

        for j in range(1, self.max_len + 1):
            # print(init_value[0:10])
            next_prob = self.sess.run(self.prediction, {self.x_: init_value})
            for i in range(self.batch_size):
                if j == 1:
                    next_word = numpy.random.multinomial(1, next_prob[i][j - 1])
                    next_word = numpy.argmax(next_word)
                else:
                    next_word = numpy.argmax(next_prob[i][j - 1])

                while next_word == self.word_to_index[unknown_token]:
                    next_word = numpy.random.multinomial(1, next_prob[i][j - 1])
                    next_word = numpy.argmax(next_word)

                if j == self.max_len:
                    additional[i] = next_word
                else:
                    init_value[i][j] = next_word

        init_value = numpy.concatenate([init_value, additional], axis=1)

        for i in range(init_value.shape[0]):
            sentence = ""
            for j in range(init_value.shape[1]):
                if init_value[i][j] == self.word_to_index[sentence_end_token]:
                    sentence = sentence + " " + self.index_to_word[init_value[i][j]]
                    break
                else:
                    sentence = sentence + " " + self.index_to_word[init_value[i][j]]

            print(sentence)

    def classification(self, testx):

        pred = []
        self.prediction = self.RNN_initialize_and_prediction_fun(1, out="label")
        i = 0

        for sens in testx :

            this_len = len(sens)
            this_sens = sens + [self.dummy] * (self.max_len - this_len)
            yhat = self.sess.run(self.prediction, feed_dict={self.x_: [this_sens]})[0]
            pred = pred + [yhat[this_len-1]]
            i = i + 1
            if i % 100 == 0 :
                print(i / len(testx) * 100)

        return(pred)



