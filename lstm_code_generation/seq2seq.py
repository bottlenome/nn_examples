import numpy
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter

EOS = -1


class Encoder(chainer.Chain):
    def __init__(self,
                 vocab_size,
                 embed_size=128,
                 hidden_size=128,
                 train=True):
        super(Encoder, self).__init__(
            embed=L.EmbedID(vocab_size, embed_size, ignore_label=EOS),
            l0=L.LSTM(embed_size, hidden_size),
        )
        self.train = train

    def reset_state(self):
        self.l0.reset_state()

    def __call__(self, tags):
        x = self.embed(tags)
        y = self.l0(x)
        return y

    def encode(self, enc):
        for i in range(enc.shape[1]):
            self(enc[:, i])
        return self.l0.h


class Decoder(chainer.Chain):
    def __init__(self,
                 vocab_size,
                 embed_size=128,
                 hidden_size=128,
                 train=True):
        super(Decoder, self).__init__(
            embed=L.EmbedID(vocab_size, embed_size, ignore_label=EOS),
            l0=L.LSTM(embed_size, hidden_size),
            l1=L.Linear(hidden_size, vocab_size)
        )
        self.train = train

    def reset_state(self):
        self.l0.reset_state()

    def __call__(self, tags):
        x = self.embed(tags)
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
        self.epsilon = 0.01

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def __call__(self, enc, dec):
        self.reset_state()
        y = []
        self.decoder.l0.h = self.encoder.encode(enc)

        batch_size = enc.shape[0]
        eos = self.xp.array([EOS for _ in range(batch_size)],
                            dtype=self.xp.int32)
        t = chainer.Variable(eos)

        for i in range(dec.shape[1]):
            char = dec[:, i]
            y.append(self.decoder(t))
            t = char
        return y

    def predict(self, enc, dic):
        assert(self.train is False)
        y = []
        self.decoder.l0.h = self.encoder.encode(enc)
        eos = self.xp.array([EOS], dtype=self.xp.int32)
        t = chainer.Variable(eos)

        c = self.decoder.l0.c
        h = self.decoder.l0.h
        result = {"": {"prob": 1.0, "hidden": h, "condition": c}}
        for i in range(self.limit):
            tmp = {}
            for key in result.keys():
                if key.find('<eos>') != -1:  # <eos> case
                    tmp[key] = result[key]
                    continue
                self.decoder.l0.h = result[key]["hidden"]
                self.decoder.l0.c = result[key]["condition"]
                y = self.decoder(t)
                probability = F.softmax(y)
                c = self.decoder.l0.c
                h = self.decoder.l0.h
                prob = result[key]["prob"]
                for j in range(len(probability)):
                    tmp[key+dic[j]] = {"prob": prob * probability[0, j].data,
                                       "hidden": h,
                                       "condition": c}
            # remove low probability sentence
            total = 0.0
            all_EOS = True
            for key in tmp.keys():
                if key.find('<eos>') == -1:
                    all_EOS = False
                if tmp[key]["prob"] < self.epsilon:
                    del tmp[key]
                else:
                    total += tmp[key]["prob"]
            if all_EOS:
                break
            # normalize probability
            for key in tmp.keys():
                tmp[key]["prob"] = tmp[key]["prob"] / total

            result = tmp
        return result


class Seq2SeqUpdater(training.StandardUpdater):
    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()

        enc, dec = self.converter(batch, self.device)

        y = optimizer.target(enc, dec)

        assert(len(y) == dec.shape[1])
        loss = 0
        for i in range(len(y)):
            loss += F.softmax_cross_entropy(y[i], dec[:, i])

        reporter.report({'loss': loss}, optimizer.target)
        # reporter.report({'accuracy': accuracy}, optimizer.target)

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


def compute_perplexity(result):
    result['perplexity'] = numpy.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = numpy.exp(result['validation/main/loss'])


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
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    import pickle
    f = open('train.pickle', 'r')
    data = pickle.load(f)
    f.close()

    model = Seq2Seq(128, 128)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    train = data.train[:-10]
    val = data.train[-10:]

    """
    model = Seq2Seq(10, 10)
    x = [[1, -1], [3, 3, -1], [2, -1, 0]]
    y = [[1, 2], [3, 3], [2, -1]]
    train = []
    for i in range(len(x)):
        train.append((numpy.array(x[i], dtype=numpy.int32),
                      numpy.array(y[i], dtype=numpy.int32)))
    """

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, 1, repeat=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = Seq2SeqUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()
    eval_model.train = False
    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=args.gpu))

    interval = 10
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity',
                               'main/accuracy', 'validation/main/accuracy']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    frequency = 1000
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    model.train = False
    dic = [ord(i) for i in range(128)]
    print(data.enc[-1])
    print(data.dec[-1])
    ret = model.predict(numpy.array(data.train[-1][0], dtype=numpy.int32), dic)
    print(ret)
