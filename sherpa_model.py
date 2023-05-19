import numpy as np
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from time import time
import pickle
import os
import random
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from sklearn.preprocessing import OneHotEncoder

from utilities import performance_plots, confusionmatrix, explain

# seeds
seed_value = 2
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.RandomState(seed_value)
np.random.seed(seed_value)
context.set_global_seed(seed_value)
ops.get_default_graph().seed = seed_value
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def create_model(n_output, input_shape):
    """
    Create keras CNN model
    :param n_output: Number of output classes
    :param input_shape:  Shape of input without batch size
    :return: Model
    """
    k_size = 50  # kernel size
    ac = 'gelu'  # activation function
    reg = tf.keras.regularizers.L2(l=0.01)
    ini = tf.keras.initializers.GlorotNormal(seed=seed_value)

    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=k_size, activation=ac,
                               kernel_initializer=ini, kernel_regularizer=reg)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=k_size, activation=ac,
                               kernel_regularizer=reg)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=k_size, activation=ac,
                               kernel_regularizer=reg)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=k_size, activation=ac,
                               kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(.2, seed=seed_value)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_output, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam()
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def fit_batch(X_train, y_train, X_val, y_val, n_output, direc, i, input_shape):
    """
    Fitting function for the model
    :param X_train: Training data
    :param y_train: Training labels
    :param X_val:  Validation data
    :param y_val: Validation labels
    :param n_output: Number of output classes
    :param direc: Saving directory
    :param i: Number of the fold, needed to save checkpoint
    :param input_shape: Data input shap excluding batchsize
    :return: Model, model history
    """
    model = create_model(n_output, input_shape)
    estop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    checkpoint_filepath = direc + str(i) + '_cnn_checkpoints'+'/cnn_check'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True, verbose=0)
    num_epochs = 200
    history = model.fit(X_train, y_train, batch_size=64, epochs=num_epochs, validation_data=(X_val, y_val), verbose=2,
                        callbacks=[estop_callback, checkpoint_callback])
    return model, history


def perform_cv(X_training, y_training, k, n_output, direc, input_shape):
    """
    Perform cross validation
    :param X_training: X training and validation data before split
    :param y_training: Labels for training and validation before split
    :param k: Number of folds for the cross validation
    :param n_output: Number of output classes
    :param direc: Result directory
    :param input_shape: Data input shap excluding batchsize
    :return: Model, performance dictionary
    """
    splits = KFold(n_splits=k, shuffle=True, random_state=seed_value)
    foldperf = {}
    best_losses = []

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(X_training)))):
        print('\nFold {}'.format(fold + 1))
        X_train = X_training[train_idx]
        y_train = y_training[train_idx]
        X_val = X_training[val_idx]
        y_val = y_training[val_idx]
        model, history = fit_batch(X_train, y_train, X_val, y_val, n_output, direc, fold, input_shape)
        foldperf['fold{}'.format(fold + 1)] = history.history
        best_losses.append(np.max(history.history['val_accuracy']))

    vl_f, tl_f, va_f, ta_f = [], [], [], []
    for f in range(1, k + 1):
        tl_f.append(foldperf['fold{}'.format(f)]['loss'][-1])
        vl_f.append(foldperf['fold{}'.format(f)]['val_loss'][-1])

        ta_f.append(foldperf['fold{}'.format(f)]['accuracy'][-1])
        va_f.append(foldperf['fold{}'.format(f)]['val_accuracy'][-1])

    print('\n\nPerformance of {} fold cross validation'.format(k))
    print("Avg Train Loss: {:.3f} Avg Val Loss: {:.3f} Avg Train Acc: {:.2f} Avg Val Acc: {:.2f}".format(
        np.mean(tl_f), np.mean(vl_f), np.mean(ta_f), np.mean(va_f)))

    return model, foldperf


def test_model(model, X_test, y_test, k, direc):
    """
    Test model and save results
    :param model: Trained model
    :param X_test: Testing data
    :param y_test: Testing labels
    :param k: Number of folds
    :param direc: Result directory
    """
    res_losses = []
    res_accs = []
    names = ['loss', 'acc']
    for i in range(k):
        checkpoint_filepath = direc + str(i) + '_cnn_checkpoints' + '/cnn_check'
        model.load_weights(checkpoint_filepath)
        results = model.evaluate(X_test, y_test, verbose=0)
        res_losses.append(results[0])
        res_accs.append(results[1])
        eval_dict = dict(zip(names, results))
        print(f"Fold {i}: {eval_dict}")
    avg_res = (np.mean(res_losses), np.mean(res_accs))
    eval_dict = dict(zip(names, avg_res))
    print(f"Avg:  {eval_dict}")
    p = direc + 'eval_dict.pkl'
    with open(p, 'wb') as g:
        pickle.dump(eval_dict, g)


def wrapper(direc, X, y):
    """
    Create, train and test model, use SHAP explainer on model, save SHAP values
    :param direc: Result directory
    :param X: Data
    :param y: Labels
    """
    t0 = time()
    print("X shape: ", X.shape, "y shape: ", y.shape, "n target classes (before OHE): ", np.unique(y), "\n")
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.1, random_state=seed_value)
    k = 5
    n_output = 3
    print("Training....")
    model, foldperf = perform_cv(X_training, y_training, k, n_output, direc, (768, 128))
    performance_plots(foldperf, k, direc)
    test_model(model, X_test, y_test, k, direc)
    y_test = enc.inverse_transform(y_test)
    confusionmatrix(model, X_test, y_test, direc)
    print("\nTime to train model: ", (time() - t0) / 60, "mins")

    t1 = time()
    print("\nShap explainer...")
    explainer, shap_values = explain(model, X_training, X_test)
    shap_values = np.array(shap_values)
    p = direc + 'shap_cnn.npy'
    with open(p, 'wb') as h:
        np.save(h, shap_values)
    print("\nTime to train SHAP: ", (time() - t1) / 60, "mins")


if __name__ == '__main__':
    t_start = time()
    print("\n\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)
    direc = 'results/'
    X = np.load("data/X_pc.npy")
    y = np.load("data/y_pc.npy")
    wrapper(direc, X, y)
    print("\nTime total: ", (time() - t_start) / 60, "mins")
