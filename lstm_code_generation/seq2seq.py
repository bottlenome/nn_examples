import numpy

import chainer.links as L
import chainer


class Encoder(chainer.Chain):
    def __init__(self,
                 vocab_size,
                 embed_size=128,
                 hidden_size=128,
                 train=True):
        super(Encoder, self).__init__(
            embed=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            l0=L.LSTM(embed_size, hidden_size),
        )
        self.train = train

    def reset_state(self):
        l0.reset_state()

    def __call__(self, one_hot):
        x = self.embed(one_hot)
        y = self.l0(x)
        return y


class Decoder(chainer.Chain):
    def __init__(self,
                 vocab_size,
                 embed_size=128,
                 hidden_size=128,
                 train=True):
        super(Decoder, self).__init__(
            embed=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            l0=L.LSTM(embed_size, hidden_size),
            l1=L.Linear(hidden_size, vocab_size)
        )
        self.train = train

    def reset_state(self):
        l0.reset_state()

    def __call__(self, one_hot):
        x = self.embed(one_hot)
        h0 = self.l0(x)
        y = self.l1(h0)
        return y


class Seq2Seq(chainer.Chain):
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 embed_size=128,
                 hidden_size=128,
                 limit_length=40,
                 train=True):
        super(Seq2Seq, self).__init__(
            encoder=Encoder(input_vocab_size, embed_size, hidden_size, train),
            decoder=Decoder(output_vocab_size, embed_size, hidden_size, train)
        )
        self.train = train
        self.limit = limit_length

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    # if y == None prediction mode
    def __call__(self, enc, dec):
        y = []
        batch_size = enc.shape[0]
        for char in enc:
            h = self.encoder(char)
        self.decoder.h = h
        eos = numpy.array([-1 for _ in range(batch_size)], dtype=numpy.int32)
        t = chainer.Variable(eos)
        for char in dec:
            y.append(self.decoder(t))
            t = char
        return y


if __name__ == '__main__':
    model = Seq2Seq(10, 10)
    x = numpy.array([[1, 2],
                     [3, 3],
                     [2, -1],
                     [-1, 0]], dtype=numpy.int32)
    x = chainer.Variable(x)
    y = numpy.array([[1, 2],
                     [3, 3],
                     [2, -1],
                     [-1, 0]], dtype=numpy.int32)
    y = chainer.Variable(y)
    for v in model(x, y):
        print(v.data)
