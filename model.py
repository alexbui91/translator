import numpy as np
import theano 
import theano.tensor as T
import properties as p
from layers import NetworkLayer, LSTM

floatX = theano.config.floatX

def train(bi_dict, data_source, data_target):
    dict_source, dict_target = bi_dict
    data_source_vis, data_source_un = data_source
    data_target_vis, data_target_un = data_target
    n_batches = len(data_source_vis) // p.batch_size
    # len of dictionary
    V_source = len(dict_source)
    V_target = len(dict_target)
    #x is the matrix with list of indices over batch_size
    x = T.matrix('x')
    y = T.matrix('y')
    index = T.lscalar()
    x_one_hot = generate_one_hot(x, V_source)
    y_one_hot = generate_one_hot(y, V_target)

    # word_layer = NetworkLayer(dim=(V_source, p.features_size), name="Word_Layer")
    # lstm_word = LSTM(dim=p.features_size, number_step=p.max_len, batch_size=p.batch_size)
    epoch = 0
    updates = []
    train_model = theano.function([index], updates=updates, givens={
        x: data_source_vis[(index * p.batch_size):((index + 1) * p.batch_size)],
        y: data_target_vis[(index * p.batch_size):((index + 1) * p.batch_size)]
    })
    while(epoch < p.epochs):
        epoch += 1
        for i in xrange(n_batches):
            train_model(i)
            break
        break
            


def generate_one_hot(data, dict_len):
    inp = T.cast(data.flatten(), dtype="int32").reshape(p.batch_size, p.max_len)
    def step(x_, h_):
        # input vector is list of words indices in sentences
        # output is matrix of one hot vector
        one_hot = T.extra_ops.to_one_hot(x_, dict_len)
        return one_hot
    results, updates = theano.scan(step, sequences=[inp], name="generate_one_hot",
                                        n_steps=p.batch_size,
                                        outputs_info=[T.alloc(np.asarray((0.), dtype='int32'), p.batch_size, p.max_len, dict_len)])
    return results[0]