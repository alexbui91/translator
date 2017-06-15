import numpy as np
import theano 
import theano.tensor as T

floatX = theano.config.floatX

class NetworkLayer(object):
    def __init__(self, dim, params=None, input_values=None, name="Network"):
        #dim expect |V| * features_size
        self.dim = dim
        self.name = name
        self.input = input_values
        self.output = None
        if params: 
            self.params = params
        else:
            self.init_params()
    
    def init_params(self):
        self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=self.dim), dtype=floatX), name="W_" + self.name)
        self.b = theano.shared(value= np.zeros((self.dim[1],), dtype=floatX), name="b_" + classname)
        self.params = [self.W, self.b]

    def feed_foward(self):
        self.output = T.dot(self.input, self.W) + self.b

    def softmax(self):
        if not self.output:
            self.feed_foward()
        self.y_prob = T.nnet.softmax(self.output)

class LSTM(object):

    def __init__(self, dim, batch_size, number_step, input_values, params=None):
        self.dim = dim
        self.input = input_values
        self.batch_size = batch_size
        self.number_step = number_step
        self.output = None
        if params is None:
            self.init_params()
        else:
            self.set_params(params)

    def init_params(self):
        Wi_values = utils.ortho_weight(self.dim)
        self.Wi = theano.shared(Wi_values, name="LSTM_Wi")
        Wf_values = utils.ortho_weight(self.dim)
        self.Wf = theano.shared(Wf_values, name="LSTM_Wf")
        Wc_values = utils.ortho_weight(self.dim)
        self.Wc = theano.shared(Wc_values, name="LSTM_Wc")
        Wo_values = utils.ortho_weight(self.dim)
        self.Wo = theano.shared(Wo_values, name="LSTM_Wo")
        Ui_values = utils.ortho_weight(self.dim)
        self.Ui = theano.shared(Ui_values, name="LSTM_Ui")
        Uf_values = utils.ortho_weight(self.dim)
        self.Uf = theano.shared(Uf_values, name="LSTM_Uf")
        Uc_values = utils.ortho_weight(self.dim)
        self.Uc = theano.shared(Uc_values, name="LSTM_Uc")
        Uo_values = utils.ortho_weight(self.dim)
        self.Uo = theano.shared(Uo_values, name="LSTM_Uo")
        b_values = np.zeros((self.dim,), dtype=theano.config.floatX)
        self.bi = theano.shared(b_values, name="LSTM_bi")
        self.bf = theano.shared(b_values, name="LSTM_bf")
        self.bc = theano.shared(b_values, name="LSTM_bc")
        self.bo = theano.shared(b_values, name="LSTM_bo")
        self.params = [self.Wi, self.Ui, self.bi, self.Wf, self.Uf, self.bf, self.Wc, self.Uc, self.bc, self.Wo, self.Uo, self.bo]
    
    def set_params(self, params):
        if params is not None and len(params) is 12:
            self.params = params
            self.Wi = params[0]
            self.Ui = params[1]
            self.bi = params[2]
            self.Wf = params[3]
            self.Uf = params[4]
            self.bf = params[5]
            self.Wc = params[6]
            self.Uc = params[7]
            self.bc = params[8]
            self.Wo = params[9]
            self.Uo = params[10]
            self.bo = params[11]

    def get_params(self):
        return self.params
    
    def feed_foward(self):
        #xt, h(t-1), c(t-1)
        #scan over sequence * batch size * width
        X_shuffled = T.cast(self.input.dimshuffle(1,0,2), theano.config.floatX)
        def step(x, h_, C_):
            i = T.nnet.sigmoid(T.dot(x, self.Wi) + T.dot(h_, self.Ui) + self.bi)
            f = T.nnet.sigmoid(T.dot(x, self.Wf) + T.dot(h_, self.Uf) + self.bf)
            c = T.tanh(T.dot(x, self.Wc) + T.dot(h_, self.Uc) + self.bc)
            o = T.nnet.sigmoid(T.dot(x, self.Wo) + T.dot(h_, self.Uo) + self.bo)
            C = c * i + f * C_
            h = o * T.tanh(C)
            return h, C
        results, updates = theano.scan(step, outputs_info=[T.alloc(np.asarray((0.), dtype=theano.config.floatX), self.batch_size, self.dim),
                                                            T.alloc(np.asarray((0.), dtype=theano.config.floatX), self.batch_size, self.dim)],
                                            sequences=[X_shuffled],
                                            name="LSTM_iteration",
                                            n_steps=self.number_step)
        #get h after perform LSTMs
        return results[0]
    
    def mean_pooling_input(self, layer_input):
        #axis = 0 is x(col), = 1 is y (row),
        self.output = T.mean(layer_input, axis=0)

