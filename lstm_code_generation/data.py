import numpy

EOS = -1


class Functions():

    def convert_to_tag_reverse(self, sentences, max_len=20):
        x = []
        for s in sentences:
            tags = []
            for i in range(len(s)):
                tags.insert(0, ord(s[i]))
            tags.insert(0, EOS)
            x.append(tags)
        return x

    def convert_to_tag(self, sentences, max_len=200):
        x = []
        for s in sentences:
            tags = []
            for i in range(len(s)):
                tags.append(ord(s[i]))
            tags.append(EOS)
            x.append(tags)
        return x

    def convert_to_tag(self, enc, dec, enc_len=50, dec_len=200, fill=False):
        x = []  # encode words(function decraration)
        y = []  # decode words(function definition)
        for e, d in zip(enc, dec):
            if len(e) >= enc_len:
                continue
            if len(d) >= dec_len:
                continue
            x_tags = []
            for i in range(len(e)):
                x_tags.insert(0, ord(e[i]))
            x_tags.append(EOS)
            if fill:
                for i in range(len(x_tags), enc_len, 1):
                    x_tags.append(EOS)
            y_tags = []
            for i in range(len(d)):
                y_tags.append(ord(d[i]))
            y_tags.append(EOS)
            if fill:
                for i in range(len(y_tags), dec_len, 1):
                    y_tags.append(EOS)
            x.append(x_tags)
            y.append(y_tags)
        return x, y

    def __init__(self, enc, dec):
        assert(len(enc) == len(dec))
        self.enc = enc
        self.dec = dec
        self.dic = {'\t': 1}
        for i in range(32, 127):
            self.dic[chr(i)] = i - 32 + 2
        self.dic['<eos>'] = 127
        x, y = self.convert_to_tag(enc, dec)
        # x = self.convert_to_tag_reverse(enc)
        # y = self.convert_to_tag(dec)

        train = []
        for i in range(len(x)):
            train.append((numpy.array(x[i], dtype=numpy.int32),
                          numpy.array(y[i], dtype=numpy.int32)))

        self.train = train


if __name__ == '__main__':
    x = ["hogeee", "mogeee", "a"*49]
    y = ["hageee", "ugeee", "b"*199]
    f = Functions(x, y)
    print(f.enc)
    print(f.dec)
    print(f.dic)
    print(f.train)
    print(f.train[2][0].shape)
    assert(f.train[2][0].shape[0] == 50)
    assert(f.train[2][1].shape[0] == 200)
