import tensorflow as tf
import RNN_for_GAN as RGG
import RNN_for_GAN_Des as RGD
import numpy

class GAN :

    def __init__(self, G_nlayer, D_nlayer, dictionary, max_len, min_len, batch_size, G_p_dropout, D_p_dropout, dim_z):

        self.G_nlayer = G_nlayer
        self.D_nlayer = D_nlayer
        self.dictionary = dictionary
        self.max_len = max_len
        self.min_len = min_len
        self.batch_size = batch_size
        self.G_p_dropout = G_p_dropout
        self.D_p_dropout = D_p_dropout

        self.Gen_model = RGG.Generator(dim_z=dim_z, nlayer=self.G_nlayer, dictionary=self.dictionary,
                                       max_len=self.max_len, min_len=self.min_len, batch_size=self.batch_size,
                                       p_dropout=self.G_p_dropout)
        self.Dis_model = RGD.Discriminator(nlayer=self.D_nlayer, dictionary=self.dictionary,
                                       max_len=self.max_len, min_len=self.min_len, batch_size=self.batch_size,
                                       p_dropout=self.G_p_dropout)

    def cost_from_fake_input(self):
        self.Dis_model.discriminate_from_generator_inputs(self.Gen_model)

        cross_entrophy = 1 * tf.log(self.Dis_model.prob_fake)
        cross_entrophy = -tf.reduce_sum(cross_entrophy, reduction_indices=2)
        cross_entrophy = tf.reshape(tf.concat(axis=1, values=cross_entrophy), [-1, ])

        cross_entrophy = tf.reduce_sum(cross_entrophy * self.mask_for_fake_)
        cross_entrophy = cross_entrophy / self.batch_size

        print("fake complete")

        return(cross_entrophy)

    def cost_from_real_input(self):
        self.Dis_model.discrimiate_from_real_data_inputs()

        cross_entrophy = tf.log(self.Dis_model.prob_real)
        cross_entrophy = -tf.reduce_sum(cross_entrophy, reduction_indices=2)
        cross_entrophy = tf.reshape(tf.concat(axis=1, values=cross_entrophy), [-1, ])

        cross_entrophy = tf.reduce_sum(cross_entrophy * self.mask_for_real_)
        cross_entrophy = cross_entrophy / self.batch_size

        print("real complete")

        return (cross_entrophy)

    def cost_of_gan_model(self):
        self.mask_for_fake_ = tf.placeholder(dtype=tf.float32, shape=(self.batch_size * self.max_len, ))
        self.mask_for_real_ = tf.placeholder(dtype=tf.float32, shape=(self.batch_size * self.max_len,))

        cost = (self.cost_from_fake_input() + self.cost_from_real_input())/2
        return cost

    def get_feed_dict(self, x_train, len_fake, sen_len_real):

        this_mask_for_fake = [0] * (len_fake - 1)
        this_mask_for_fake = this_mask_for_fake + [1]
        this_mask_for_fake = this_mask_for_fake + [0] * (self.max_len - len_fake)

        mask_for_fake = self.batch_size * this_mask_for_fake

        mask_for_real = []

        for i in sen_len_real :
            this_mask_for_real = [0] * (i-1)
            this_mask_for_real = this_mask_for_real + [1]
            this_mask_for_real = this_mask_for_real + [0] * (self.max_len - i)
            mask_for_real = mask_for_real + this_mask_for_real

        return({self.Dis_model.x_ : x_train, self.Gen_model.n_:len_fake, self.mask_for_fake_ : mask_for_fake, self.mask_for_real_ : mask_for_real})



