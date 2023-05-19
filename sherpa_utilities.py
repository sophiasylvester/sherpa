import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def performance_plots(foldperf, k, direc):
    """
    Create and save performance plots
    :param foldperf: Performance dictionary
    :param k: Number of folds
    :param direc: Results directory
    """
    train_colors = ['cornflowerblue', 'royalblue', 'deepskyblue', 'dodgerblue', 'blue']
    val_colors = ['yellow', 'gold', 'orange', 'darkorange', 'orangered']
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Performance of " + direc[:-1])
    ax0 = fig.add_subplot(121, title='Loss')
    ax1 = fig.add_subplot(122, title='Accuracy')

    for i in range(k):
        fold = str(i+1)
        ax0.plot(foldperf['fold{}'.format(i+1)]['loss'], label='Train_'+fold, color=train_colors[i])
        ax0.plot(foldperf['fold{}'.format(i+1)]['val_loss'], label='Val_'+fold, color=val_colors[i])
        ax1.plot(foldperf['fold{}'.format(i+1)]['accuracy'], label='Train_'+fold, color=train_colors[i])
        ax1.plot(foldperf['fold{}'.format(i+1)]['val_accuracy'], label='Val_'+fold, color=val_colors[i])
    ax0.set_ylabel('Loss')
    ax0.set_xlabel('Epochs')
    ax0.legend()
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    p = direc + "perform_plot.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")


def confusionmatrix(model, X_test, y_test, direc):
    """
    Create and save confusion matrix
    :param model: Model
    :param X_test: Testing data
    :param y_test: Testing labels
    :param direc: Result directory
    """
    class_names = ['face', 'blurred', 'scrambled']
    y_pred = model.predict(X_test)
    y_pred = tf.concat(y_pred, axis=0)
    y_pred = tf.argmax(y_pred, axis=1)
    y_pred = y_pred.numpy()

    fig, ax = plt.subplots()
    cm = tf.math.confusion_matrix(y_test, y_pred)
    cm = cm.numpy()
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cmn, annot=True, fmt='.2f')
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    p = direc + "cmplot.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")


def explain(model, X_train, X_test):
    """
    Apply SHAP gradient explainer
    :param model: Model
    :param X_train: Training data
    :param X_test: Testing data
    :return: Explainer, SHAP values
    """
    explainer = shap.GradientExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values
