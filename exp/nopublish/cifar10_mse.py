import argparse
import keras
from keras.datasets import cifar10
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from sklearn.metrics import mean_squared_error
import time

keras.backend.set_image_data_format("channels_last")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trees", dest="n_trees", type=int, default=1000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    assert x_train.shape[-1] == 3
    assert x_test.shape[-1] == 3
    x_train = x_train.reshape((x_train.shape[0], -1, 3))
    x_test = x_test.reshape((x_test.shape[0], -1, 3))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    print("[mnist] x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
    print("[mnist] x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))

    n_trees = args.n_trees
    print("n_trees={}".format(args.n_trees))

    # the autoencoder results
    for mi, model in enumerate(("supervised", "unsupervised")):
        print("Start Autoencoder using {} model".format(model))
        eforest_channels = []
        start = time.time()
        if model == "supervised":
            for c in range(3):
                eforest = RandomForestClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, random_state=0)
                eforest.fit(x_train[:, :, 0], y_train)
                eforest_channels.append(eforest)
        else:
            for c in range(3):
                eforest = RandomTreesEmbedding(n_estimators=n_trees, max_depth=None, n_jobs=-1, random_state=0)
                eforest.fit(x_train[:, :, 0])
                eforest_channels.append(eforest)
        print("cost/fit={}".format(time.time() - start))

        start = time.time()
        x_encode_channels = []
        for c in range(3):
            x_encode = eforest.encode(x_test[:, :, c])
            x_encode_channels.append(x_encode)
        print("cost/encode={}".format(time.time() - start))

        start = time.time()
        x_decode_channels = []
        for c in range(3):
            x_decode = eforest.decode(x_encode_channels[c])
            x_decode_channels.append(x_decode)
        print("cost/decode={}".format(time.time() - start))

        decodes = np.stack(x_decode_channels, axis=-1)
        mse = mean_squared_error(x_test.reshape(-1), decodes.reshape(-1))
        print("mse: {}".format(np.mean(mse)))
