import tensorflow as tf

class Discriminator :

    def __init__(self, nlayer, dictionary, max_len, min_len, batch_size, p_dropout):

        self.nlayer = nlayer
        self.dictionary = dictionary
        self.max_len = max_len
        self.min_len = min_len
        self.batch_size = batch_size
        self.dummy = len(dictionary) + 1

        d_out_cells = []

        for i in range(self.nlayer) :
            cell = tf.nn.rnn_cell.BasicLSTMCell(2)
            d_out_cells = d_out_cells + [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - p_dropout)]

        self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(d_out_cells)


    def discriminate_from_generator_inputs(self, gen_model):

        init_state = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        x_one_hot = tf.one_hot(gen_model.prediction_dummied, depth=len(self.dictionary))

        with tf.variable_scope("discriminator"):
            self.logit_fake, state = tf.nn.dynamic_rnn(cell=self.rnn_cell, inputs=x_one_hot, initial_state=init_state,
                                                  dtype=tf.float32)

        self.prob_fake = tf.nn.softmax(self.logit_fake)
        self.prediction_fake = tf.argmax(input=self.prob_fake, axis=2)


    def discrimiate_from_real_data_inputs(self):

        init_state = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        self.x_ = tf.placeholder(tf.int64, shape=[None, self.max_len])
        x_one_hot = tf.one_hot(self.x_, depth=len(self.dictionary))

        with tf.variable_scope("discriminator", reuse=True):
            self.logit_real, state = tf.nn.dynamic_rnn(cell=self.rnn_cell, inputs=x_one_hot, initial_state=init_state,
                                                       dtype=tf.float32)

        self.prob_real = tf.nn.softmax(self.logit_real)
        self.prediction_real = tf.argmax(input=self.prob_real, axis=2)