import numpy
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training

EOS = 0


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
        for i in range(dec.shape[1]):
            char = dec[:, i]
            y.append(self.decoder(t))
            t = char
        return y


class Seq2SeqUpdater(training.StandardUpdater):
    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()

        enc, dec = self.converter(batch, self.device)

        y = optimizer.target(enc, dec)

        print(y)
        print(dec)
        assert(len(y) == dec.shape[1])
        loss = 0
        for i in range(len(y)):
            loss += F.softmax_cross_entropy(y[i], dec[:, i])

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    model = Seq2Seq(10, 10)
    x = [[1, 2, -1], [3, 3, -1], [2, -1, 0]]
    y = [[1, 2], [3, 3], [2, -1]]
    train = []
    for i in range(len(x)):
        train.append((numpy.array(x[i], dtype=numpy.int32),
                      numpy.array(y[i], dtype=numpy.int32)))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = Seq2SeqUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.run()
