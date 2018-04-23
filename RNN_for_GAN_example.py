import RNN_for_GAN as RG1
import RNN_for_GAN_Des as RG2
import GAN_agg as myGAN
import Data_Preprocessor as DP
import numpy

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

# Gen_fun = RG1.Generator(dim_z=6, nlayer=2, dictionary=preprocessor.index_to_word, max_len=preprocessor.max_len, min_len=preprocessor.min_len, batch_size= 15, p_dropout=0.1)

# Dis_fun = RG2.Discriminator(nlayer = 2, dictionary= preprocessor.index_to_word, max_len = preprocessor.max_len, min_len= preprocessor.min_len, batch_size= 15, p_dropout=0.1)
# Dis_fun.discriminate_from_generator_inputs(Gen_fun)
# Dis_fun.discrimiate_from_real_data_inputs(x)

GAN = myGAN.GAN(G_nlayer=2, D_nlayer=2, dictionary=preprocessor.index_to_word, max_len=preprocessor.max_len, min_len=preprocessor.min_len, batch_size= 20, G_p_dropout=0.1, D_p_dropout=0.1, dim_z = 5)
preprocessor.dummy_generation()

cost = GAN.cost_of_gan_model()
feed_dict = GAN.get_feed_dict(x_train=preprocessor.x_train[0:20], len_fake=10, sen_len_real=preprocessor.len_sens[0:20])
print(cost)
print(feed_dict)
import tensorflow as tf

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(cost, feed_dict=feed_dict))

