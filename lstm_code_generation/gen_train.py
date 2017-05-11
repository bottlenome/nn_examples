import argparse
import os
import astor
import ast
import re
import pickle
import data


def get_functions(filepath):
    enc = []
    dec = []
    with open(filepath) as f:
        source = f.read()
    tree = ast.parse(source)
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.FunctionDef):
            function_source = astor.to_source(stmt)
            function_def = function_source.split('\n')[0] + '\n'
            function_des = function_source[len(function_def):]
            enc.append(function_def)
            dec.append(function_des)
    return enc, dec


def get_traindatas(path):
    enc = []
    dec = []
    for (root, dirs, files) in os.walk(path):
        for f in files:
            if re.match(r".*\.py$", f):
                e, d = get_functions(root + os.sep + f)
                enc += e
                dec += d
    return data.Functions(enc, dec)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--path', '-p', default='./',
                        help='target source directory')
    parser.add_argument('--filename', '-f', default='train.pickle',
                        help='learn target text')
    args = parser.parse_args()

    ret = get_traindatas(args.path)
    f = open(args.filename, 'w')
    pickle.dump(ret, f)
    f.close()


if __name__ == '__main__':
    main()
