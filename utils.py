import theano
import theano.tensor as T
import pickle
import numpy as np
import properties
import os.path as path
import codecs


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)


def Tanh(x):
    y = T.tanh(x)
    return(y)


def Iden(x):
    y = x
    return(y)


def dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output


def find_largest_number(num1, num2, num3):
    largest = num1
    if (num1 >= num2) and (num1 >= num3):
       largest = num1
    elif (num2 >= num1) and (num2 >= num3):
       largest = num2
    else:
       largest = num3
    return largest


def save_file(name, obj, utf8=False):
    if not utf8:
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with codecs.open(name, 'w', 'utf-8') as f:
             f.write(obj)


def load_file(pathfile, utf8=False):
    if not path.exists(pathfile):
        return None
    if not utf8:
        with open(pathfile, 'rb') as f:
            data = pickle.load(f)
        return data 
    else:
        with codecs.open('aa.txt', 'r', 'utf-8') as f:
            data = f.read()
        return data 


def load_file_lines(pathfile):
    if path.exists(pathfile):
        with open(pathfile, 'rb') as f:
            return f.readlines()


def get_num_words(vocabs, sent):
    length = 0
    words = sent.split()
    for word in words:
        if word in vocabs:
            length += 1
    return length


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def check_array_full_zeros(arr):
    for x in arr:
        if not x:
            return False
    return True

def save_layer_params(layers, name):
    # now = time.time()
    params = [param.get_value() for param in layers.params]
    save_file('%s.txt' % name, params)