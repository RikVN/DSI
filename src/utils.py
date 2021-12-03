'''Simply functions that we re-use often'''

import sys
import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.nl import Dutch


def write_to_file(lst, out_file):
    '''Write list to file'''
    with open(out_file, "w") as out_f:
        for line in lst:
            out_f.write(line.strip() + '\n')
    out_f.close()


def setup_spacy_tokenizer(lang):
    '''Given a language code, return the correct tokenizer'''
    if lang == "en":
        nlp = English()
    elif lang == "es":
        nlp = Spanish()
    elif lang == "nl":
        nlp = Dutch()
    else:
        raise ValueError("Language {0} not supported so far".format(lang))
    return nlp.tokenizer


def load_json_dict(d):
    '''Funcion that loads json dictionaries'''
    with open(d, 'r') as in_f:
        dic = json.load(in_f)
    in_f.close()
    return dic


def read_dsi_corpus(corpus_file):
    '''Read in document and label (tab-separated)'''
    documents, labels = [], []
    for line in open(corpus_file, 'r'):
        l = line.strip().split('\t')
        if len(l) > 2:
            raise ValueError("Data lines should only contain 1 tab:\n {0}".format(line.strip()))
        documents.append(l[0])
        labels.append(l[1])
    return documents, labels


# Taken directly from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def plot_confusion_matrix(cm,
                          target_names,
                          save_to,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    #accuracy = np.trace(cm) / np.sum(cm).astype('float')

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_to, bbox_inches = "tight")
