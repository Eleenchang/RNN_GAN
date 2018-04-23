import tensorflow as tf

class Generator :

    def __init__(self, dim_z, nlayer, dictionary, max_len, min_len, batch_size, p_dropout):

        self.dim_z = dim_z
        self.nlayer = nlayer
        self.dictionary = dictionary
        self.max_len = max_len
        self.min_len = min_len
        self.dummy = len(dictionary) + 1

        self.n_ = tf.placeholder(dtype = tf.int32)
        self.z = tf.random_normal((batch_size, self.n_, self.dim_z))

        d_out_cells = []

        for i in range(self.nlayer) :
            cell = tf.nn.rnn_cell.BasicLSTMCell(len(self.dictionary))
            d_out_cells = d_out_cells + [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - p_dropout)]

        self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(d_out_cells)

        init_state = self.rnn_cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope("generator"):
            self.logit, state = tf.nn.dynamic_rnn(cell=self.rnn_cell, inputs=self.z, initial_state=init_state, dtype=tf.float32)

        self.prob = tf.nn.softmax(self.logit)
        self.prediction = tf.argmax(input = self.prob, axis=2)
        self.prediction_dummied = tf.concat([self.prediction, self.dummy*tf.ones(shape = (batch_size, self.max_len - self.n_), dtype = tf.int64)], 1)

