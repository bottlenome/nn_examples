import numpy

EOS = -1


class Functions():

    def convert_to_tag_reverse(self, sentences):
        x = []
        for s in sentences:
            tags = []
            for i in range(len(s)):
                tags.insert(0, ord(s[i]))
            tags.insert(0, EOS)
            x.append(tags)
        return x

    def convert_to_tag(self, sentences):
        x = []
        for s in sentences:
            tags = []
            for i in range(len(s)):
                tags.append(ord(s[i]))
            tags.append(EOS)
            x.append(tags)
        return x

    def __init__(self, enc, dec):
        assert(len(enc) == len(dec))
        self.enc = enc
        self.dec = dec
        self.dic = ['\t']
        for i in range(32, 127):
            self.dic.append(chr(i))
        self.dic.append('<eos>')
        x = self.convert_to_tag_reverse(enc)
        y = self.convert_to_tag(dec)

        train = []
        for i in range(len(x)):
            train.append((numpy.array(x[i], dtype=numpy.int32),
                          numpy.array(y[i], dtype=numpy.int32)))

        self.train = train


if __name__ == '__main__':
    x = ["hogeee", "mogeee"]
    y = ["hageee", "ugeee"]
    f = Functions(x, y)
    print(f.enc)
    print(f.dec)
    print(f.dic)
    print(f.train)
