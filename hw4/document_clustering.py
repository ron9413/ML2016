from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import pandas as pd

def parser():
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--directory",
                  dest="data_path", type="string",
                  help="directory that contains title_StackOverflow.txt and check_index.csv")
    op.add_option("--output",
                  dest="outfile", type="string",
                  help="output.csv")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=False,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")

    (opts, args) = op.parse_args()

    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    return opts

def tf_idf(opts):
    with open(opts.data_path + 'title_StackOverflow.txt', 'r') as f:
        dataset = [title for title in f.readlines()]

    my_words = set(['does', 'list', 'error', 'use', 'using', 'way', 'problem', 'question', 'method'
                    '2008', '2007', '2005', '10', 'file', 'files', 'custom', 'function'])
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)

    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=set(my_stop_words),
                                 use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    return X, vectorizer

def lsa(opts, X):
    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()

        return X, svd


def kmeans(opts, X, svd, vectorizer, n_clusters=20):
    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=30,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=30,
                    verbose=opts.verbose)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    if not opts.use_hashing:
        print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(n_clusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

    return km

def predict(opts, km):
    check_index = pd.read_csv(opts.data_path + 'check_index.csv')
    check_index = np.asarray(check_index)
    predict = km.labels_[check_index[:, 1:]]
    return predict

def write_file(opts, predict):
    with open(opts.outfile, 'w') as f:
        f.write('ID,Ans\n')
        for i in range(len(predict)):
            f.write(str(i) + ',')
            if predict[i][0] == predict[i][1]:
                f.write('1\n')
            else:
                f.write('0\n')

if __name__ == "__main__":
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    opts = parser()
    X, vectorizer = tf_idf(opts)
    X, svd = lsa(opts, X)
    km = kmeans(opts, X, svd, vectorizer, 85)
    prediction = predict(opts, km)
    write_file(opts, prediction)




