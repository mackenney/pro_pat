from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier


# ===================================================================
# KERAS NEURAL NETWORK CLASSIFIER

def baseline_estimator(input_dim, activation='relu'):
    def base_model():
        model = Sequential()
        model.add(Dense(100, input_dim=input_dim, activation=activation))
        # model.add(Dense(50, activation=activation))
        model.add(
            Dense(7, activation='softmax'))  # 7 classes, with softmax activation
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    # construct the estimator
    estimator = KerasClassifier(
        build_fn=base_model,
        epochs=100,
        batch_size=5,
        verbose=1
    )
    return estimator


def fitted_classifier_NN(X_tr, y_tr, activation='relu'):
    """
    Return only the trained estimator... it has the methods of any
    estimator such as .predict(X_te), .score(X_te, y_te)
    :param X_tr: training features
    :param X_te: testing features
    :param activation: activation function to use in the hidden layers
    :return: TRAINED MODEL FROM KERAS
    """
    estimator = baseline_estimator(X_tr.shape[1], activation)
    estimator.fit(X_tr, y_tr)
    return estimator


def training_and_classification_NN(X_tr, X_te, y_tr, y_te, activation='relu'):
    """
    Returns the results (predicted classes) and score
    :param X_tr: training features
    :param X_te: testing features
    :param y_tr: training labels (classes)
    :param y_te: testing labels (classes)
    :param activation: activation function to use in the hidden layers
    :return:
    """
    estimator = fitted_classifier_NN(X_tr, y_tr, activation=activation)

    results = estimator.predict(X_te)

    score = estimator.score(X_te, y_te)
    print('accuracy:', score)
    return results, score

# ===================================================================
# KNN CLASSIFIER


def fitted_classifier_KNN(X_tr, y_tr, neighbors_param=3):
    """
    :param X_tr: training features
    :param X_te: testing features
    :return: TRAINED MODEL FOR KNN CLASSIFICATION
    """
    knn = KNeighborsClassifier(n_neighbors=neighbors_param, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    return knn


def training_and_classification_knn(X_tr, X_te, y_tr, y_te, neighbors_param=3):
    """
    KNN classification
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :param neighbors_param:
    :return:
    """
    knn = fitted_classifier_KNN(X_tr, y_tr, neighbors_param=neighbors_param)
    results = knn.predict(X_te)
    print('accuracy: ', knn.score(X_te, y_te))
    return results


# ====================================================================
# SVM ClASSIFIER


def fitted_classifier_SVM(X_tr, y_tr, kernel='linear', _C=1, degree=3):
    """
    :param X_tr: training features
    :param X_te: testing features
    :return: TRAINED MODEL FOR SVM CLASSIFICATION
    """
    if kernel == 'poly':
        svm = SVC(kernel=kernel, C=_C, degree=degree)
    else:
        svm = SVC(kernel=kernel, C=_C)
    svm.fit(X_tr, y_tr)
    return svm


def training_and_classification_SVM(X_tr, X_te, y_tr, y_te, kernel='linear', _C=1, degree=3):
    """
    Support Vector Machine Classificator, RBF kernel
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :return: prediction for X_te
    """
    svm = fitted_classifier_SVM(X_tr, y_tr, kernel=kernel, _C=_C, degree=degree)
    print('accuracy:', svm.score(X_te, y_te))
    return svm.predict(X_te)

# ====================================================================
# LDA ClASSIFIER


def fitted_classifier_LDA(X_tr, y_tr, solver='lsqr'):
    """
    :param X_tr: training features
    :param X_te: testing features
    :return: TRAINED MODEL FOR LDA CLASSIFICATION
    """
    if solver == 'svd':
        lda = LDA(solver=solver)
    else:
        lda = LDA(solver=solver, shrinkage='auto')
    lda.fit(X_tr, y_tr)
    return lda


def training_and_classification_LDA(X_tr, X_te, y_tr, y_te, solver='lsqr'):
    """
    Linear Discriminant Analysis classification
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :param solver:
    :return:
    """
    lda = fitted_classifier_LDA(X_tr, y_tr, solver=solver)
    results = lda.predict(X_te)
    score = lda.score(X_te, y_te)
    print('accuracy:', score)
    return results, score

# ====================================================================
# SKLEARN MULTI LAYER PERCEPTRON CLASSIFIER


def fitted_classifier_MLP(X_tr, y_tr, solver='lbfgs'):
    """
    :param X_tr: training features
    :param X_te: testing features
    :return: TRAINED MODEL FOR KNN CLASSIFICATION
    """
    mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic',
                        solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.001,
                        power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False,
                        warm_start=False, momentum=0.9, nesterovs_momentum=True,
                        early_stopping=False, validation_fraction=0.1,
                        beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mlp.fit(X_tr, y_tr)
    return mlp


def training_and_classification_MLP(X_tr, X_te, y_tr, y_te, solver='lbfgs'):
    """
    Multi-layer Perceptron classifier
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    :param X_tr:
    :param X_te:
    :param y_tr:
    :param y_te:
    :param solver:
    :return:
    """
    # mlp = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp = fitted_classifier_MLP(X_tr, y_tr, solver=solver)
    results = mlp.predict(X_te)
    score = mlp.score(X_te, y_te)
    print('accuracy:', score)
    return results, score


# ==================================================================
# CLASSIFICATION ONLY

def classify(classifier, X_te, y_te):
    """
    :param classifier: estimator with methods .predict, .score, etc
    :param X_te: testing features
    :param y_te: testing correct labels (classes)
    :return: results and score
    """
    results = classifier.predict(X_te)
    score = classifier.score(y_te)
    print('accuracy:', score)
    return results, score
