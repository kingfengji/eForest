import argparse
from keras.datasets import mnist
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from sklearn.metrics import mean_squared_error
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trees", dest="n_trees", type=int, default=1000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    print("[mnist] x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
    print("[mnist] x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))

    n_trees = args.n_trees
    print("n_trees={}".format(args.n_trees))

    for mi, model in enumerate(("supervised", "unsupervised")):
        print("Start Autoencoder using {} model".format(model))
        start = time.time()
        if model == "supervised":
            eforest = RandomForestClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, random_state=0)
            eforest.fit(x_train, y_train)
        else:
            eforest = RandomTreesEmbedding(n_estimators=n_trees, max_depth=None, n_jobs=-1, random_state=0)
            eforest.fit(x_train)
        print("cost/fit={:.3f}".format(time.time() - start))

        start = time.time()
        x_encode = eforest.encode(x_test)
        print("cost/encode={:.3f}".format(time.time() - start))

        start = time.time()
        x_decode = eforest.decode(x_encode)
        print("cost/decode={:.3f}".format(time.time() - start))

        mse = mean_squared_error(x_test.reshape(-1), x_decode.reshape(-1))
        print("mse: {}".format(np.mean(mse)))
