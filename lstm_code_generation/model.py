import chainer
import chainer.links as L
import numpy as np
import chainer.cuda as cuda
from chainer import Variable


class MyNLSTM(L.NStepLSTM):
    def __init__(self,
                 in_size,
                 out_size,
                 batch_size,
                 use_cudnn,
                 n_layers=1,
                 dropout=0.5,
                 train=True):
        super(MyNLSTM, self).__init__(n_layers, in_size, out_size,
                                      dropout, use_cudnn)
        self.batch_size = batch_size
        self.train = train
        self.reset_state()

    def set_state(self, cx, hx):
        assert(isinstance(cx, chainer.Variable))
        assert(isinstance(hx, chainer.Variable))
        assert(self.cx.shape == cx.shape)
        assert(self.hx.shape == hx.shape)
        self.cx = cx
        self.hx = hx
        if lsef.xp == np:
            self.cx.to_cpu()
            self.hx.to_cpu()
        else:
            self.cx.to_gpu()
            self.hx.to_gpu()

    def reset_state(self):
        self.hx = chainer.Variable(
                    self.xp.zeros(
                        (self.n_layers, self.batch_size, self.out_size),
                        dtype='float32'), volatile='auto')
        self.cx = self.hx

    def __call__(self, xs):
        hy, cy, ys = super(MyNLSTM, self).__call__(
                        self.hx, self.cx, xs, self.train)
        self.hy = hy
        self.cy = cy
        return ys


class Encoder(chainer.Chain):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 use_cudnn,
                 n_lstm_layer=1,
                 dropout=0.5,
                 train=True):
        super(Encoder, self).__init__(
            embed=L.EmbedID(vocab_size, embed_size),
            l0=L.NStepLSTM(n_lstm_layer, embed_size, hidden_size,
                           dropout, use_cudnn=use_cudnn),
        )
        self.train = train
        self.cx = None
        self.hx = None

    def __call__(self, tags):
        x = self.embed(tags)
        hy, cy, y = self.l0(self.hx, self.cx, x)
        self.cx = cy
        self.hx = hy
        return y

    def encode(self, enc):
        for i in range(enc.shape[1]):
            self(enc[:, i])
        return self.l0.h


if __name__ == '__main__':
    model = MyNLSTM(2, 2, 2, False)
    a = np.array([[1, 2], [2, 3]], dtype=np.float32)
    b = np.array([[3, 4]], dtype=np.float32)
    c = np.array([[3, 4]], dtype=np.float32)
    x = [a, b]
    y = model(x)
    print(y[0].data)
    print(y[1].data)
