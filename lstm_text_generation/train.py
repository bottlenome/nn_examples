import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import datasets

import argparse
import numpy as np


class Generator(chainer.Chain):
    def __init__(self, n_sequence, n_chars, n_units, train=False):
        super(Generator, self).__init__(
            l0=L.LSTM(n_sequence * n_chars, n_units),
            l1=L.Linear(n_units, n_chars),
        )
        self.train = train

    def __call__(self, x):
        h0 = F.dropout(self.l0(x), train=self.train)
        y = F.dropout(F.relu(self.l1(h0)), train=self.train)
        return y

    def reset_state(self):
        self.l0.reset_state()


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=60,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--path', '-p', default='./nietzsche.txt',
                        help='learn target text')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='Number of units')
    parser.add_argument('--model', '-m', default='model.npz',
                        help='Model file name to serialize')
    parser.add_argument('--train_size', '-t', type=int, default=-1,
                        help='train sequence size')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # initialize data
    text = open(args.path).read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.float32)
    y = np.zeros(len(sentences), dtype=np.int32)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i] = char_indices[next_chars[i]]

    train = chainer.datasets.TupleDataset(X, y)
    train, test = datasets.split_dataset_random(train, len(train) - 1000)
    train, val = datasets.split_dataset_random(train, len(train) - 1000)
    if args.train_size == -1:
        pass
    else:
        train = train[:args.train_size]

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, 1, repeat=False)
    test_iter = chainer.iterators.SerialIterator(test, 1, repeat=False)

    model = L.Classifier(Generator(maxlen, len(chars), args.unit))

    # Set up optimizer
    optimizer = chainer.optimizers.RMSprop()
    optimizer.setup(model)

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, maxlen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=args.gpu,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

    interval = 10
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity',
                               'main/accuracy', 'validation/main/accuracy']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # Serialize the final model
    chainer.serializers.save_npz(args.model, model)


if __name__ == '__main__':
    main()
