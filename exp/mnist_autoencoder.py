import argparse
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding


def plot_mnist(rheads, datas):
    """
    datas: ndarray
        shape = [n_rows, 10, 784]
    """
    n_rows = len(rheads)
    n_cols = len(datas[0])

    fig = plt.figure()
    for r in range(n_rows):
        fig.add_subplot(n_rows * 2, n_cols, r * 2 * n_cols + 0 + 1)
        plt.text(0, 0, rheads[r])
        plt.axis("off")
        for c in range(n_cols):
            fig.add_subplot(n_rows * 2, n_cols, (r * 2 + 1) * n_cols + c + 1)
            plt.imshow(datas[r][c], cmap=plt.cm.Greys_r)
            plt.axis("off")
    return fig


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

    test_images = np.zeros((10, 28 * 28))
    for label in range(10):
        index = np.where(y_test == label)[0][0]
        test_images[label] = x_test[index]
    results = np.zeros((2, 10, 28 * 28))
    for mi, model in enumerate(("supervised", "unsupervised")):
        print("Start Autoencoder using {} model".format(model))
        if model == "supervised":
            eforest = RandomForestClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, random_state=0)
            eforest.fit(x_train, y_train)
        else:
            eforest = RandomTreesEmbedding(n_estimators=n_trees, max_depth=None, n_jobs=-1, random_state=0)
            eforest.fit(x_train)

        x_encode = eforest.encode(test_images)
        x_decode = eforest.decode(x_encode)
        results[mi] = x_decode
    rheads = ["origin", "supervised", "unsupervised"]
    test_images = test_images.reshape(1, 10, 28, 28)
    results = results.reshape(2, 10, 28, 28)
    fig = plot_mnist(rheads, np.vstack((test_images, results)))
    plt.show()

    import IPython
    IPython.embed()
