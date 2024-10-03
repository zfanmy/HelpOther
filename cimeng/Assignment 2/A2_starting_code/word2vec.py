import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from features import get_features_w2v, document_to_vector
from classifier import search_C
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

def search_hyperparams(Xt_train, y_train, Xt_val, y_val):
    """Search the best values of hyper-parameters for Word2Vec as well as the
    regularisation parameter C for logistic regression, using the validation set.

    Args:
        Xt_train, Xt_val (list(list(list(str)))): Lists of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens) for training and validation, respectively.
        y_train, y_val: Dense vectors (np.ndarray) of class labels for training
            and validation, respectively. Each element of the vector is either
            0 or 1.

    Returns:
        dict(str : union(int, float)): The best values of hyper-parameters.
    """
    # TODO: tune at least two of the many hyper-parameters of Word2Vec 
    #       (e.g. vector_size, window, negative, alpha, epochs, etc.) as well as
    #       the regularisation parameter C for logistic regression
    #       using the validation set.

    # Define different parameter values to test
    vector_size_values = [100, 200, 300]
    window_values = [3, 5, 7]
    negative_values = [5, 10, 15]
    best_params = {}
    best_acc = 0.0

    # Grid search through different combinations of hyperparameters
    for vector_size in vector_size_values:
        for window in window_values:
            for negative in negative_values:
                print(f"Training Word2Vec with vector_size={vector_size}, window={window}, negative={negative}")
                model = train_w2v(Xt_train, vector_size=vector_size, window=window, negative=negative)
                X_train_w2v = [document_to_vector(doc, model.wv) for doc in Xt_train]
                X_val_w2v = [document_to_vector(doc, model.wv) for doc in Xt_val]

                # Train a classifier (e.g., Logistic Regression) using the generated Word2Vec vectors
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train_w2v, y_train)
                y_pred = clf.predict(X_val_w2v)
                acc = accuracy_score(y_val, y_pred)
                print(f"Validation accuracy: {acc}")

                # Update best parameters if current model performs better
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'vector_size': vector_size, 'window': window, 'negative': negative}

    print(f"Best params: {best_params} with accuracy: {best_acc}")
    return best_params, best_acc


def train_w2v(Xt_train, vector_size=200, window=5, min_count=5, negative=10, epochs=3, seed=101, workers=10,
              compute_loss=False, **kwargs):
    """Train a Word2Vec model.

    Args:
        Xt_train (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        for descriptions of the other arguments.

    Returns:
        gensim.models.keyedvectors.KeyedVectors: A mapping from words (string) to their embeddings
            (np.ndarray)
    """
    sentences_train = [sent for doc in Xt_train for sent in doc]

    # TODO: train the Word2Vec model
    print(f'Training word2vec using {len(sentences_train):,d} sentences ...')
 
    # The code below needs to be modified.
    w2v_model = Word2Vec(sentences=Xt_train, vector_size=vector_size, window=window,
                         min_count=min_count, negative=negative, seed=seed, workers=workers,
                         epochs=epochs, compute_loss=compute_loss, **kwargs)
    return w2v_model.wv

