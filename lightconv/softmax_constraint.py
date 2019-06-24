from keras.constraints import Constraint
from keras.layers import Dropout
import keras.backend as K


class SoftmaxConstraint(Constraint):
    def __init__(self, rate=0.1):
        super(SoftmaxConstraint, self).__init__()
        self.rate = rate

    def __call__(self, w):
        w = K.softmax(w, 0)
        w = Dropout(self.rate)(w)
        w /= (1 - self.rate)
        return w
