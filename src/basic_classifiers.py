#!/usr/bin/env python

'''Run a basic classifier (NB or SVM) for DSI classification. Potentially print best features.'''

import sys
import argparse
import numpy as np
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from utils import read_dsi_corpus, plot_confusion_matrix, load_json_dict, write_to_file
from config import to_small, word_list_file


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Input file to use for training")
    parser.add_argument("-t", "--test_file", default='', type=str,
                        help="File we test on, if not specified do CV on train set")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-a", "--algorithm", choices = ["nb", "svm"], default="svm", type=str,
                        help="What algorithm are we using? Currently only NB or SVM")
    parser.add_argument("-cv", "--cross_validate", default=5, type=int,
                        help="How many folds for CV? Only do when no test file is added")
    parser.add_argument("-md", "--min_df", default=5, type=int,
                        help="Minimum amount a feature should occur before being added")
    parser.add_argument("-f", "--features", action="store_true",
                        help="Print best features per class")
    parser.add_argument("-owf", "--only_word_features", action="store_true",
                        help="If added, we use only the word features as defined in a dict")
    parser.add_argument("-l", "--limit_train", default=0, type=int,
                        help="Limit training set to this amount of instances (default 0 means no limit)")
    parser.add_argument("-cm", "--confusion", default='', type=str,
                        help="Save plot of confusion matrix here, if not added do not plot")
    parser.add_argument("-ovr", "--one_vs_rest", action="store_true",
                        help="Do one vs rest classification instead of one vs one (default)")
    parser.add_argument("-d", "--down_sample", default=0, type=int,
                        help="Downsample categories to this amount of instances (default 0 means no limit)")
    parser.add_argument("-ao", "--also_other", action="store_true",
                        help="Also downsample the 'other' category. The default is that other is the same size as the rest of the data combined")
    parser.add_argument("-fc", "--filter_categories", nargs="*", default=[],
                        help="Filter the given categories from the data sets, both train and test")
    parser.add_argument("-tnl", "--test_no_labels", default="",
                        help="The test set has no labels: print predictions to this file")
    parser.add_argument("-pr", "--probabilities", action="store_true",
                        help="Print the probabilities to a file instead of the labels for -tnl")
    args = parser.parse_args()
    if not args.test_file and args.cross_validate == 0:
        raise ValueError("Either specify a test set or use -cv > 0")
    if args.features and args.algorithm != "svm":
        raise ValueError("Function --features is only implemented for -a svm")
    return args


def down_sample_other(X_train, Y_train):
    '''Downsample the "other" label to be the same size as the full DSI data set without other'''
    X_final, Y_final = [], []
    # This is the amount of non-other instances
    num_dsi = len([x for x in Y_train if x != "other"])
    cur_other = 0
    # Keep all instances, except perhaps filter some of the "other" category
    for inst, label in zip(X_train, Y_train):
        if label == "other":
            cur_other += 1
            if cur_other > num_dsi:
                continue
        X_final.append(inst)
        Y_final.append(label)
    return X_final, Y_final


def down_sample_data(X_orig, Y_orig, max_inst, also_other):
    '''Downsample categories to the max amount of instances'''
    X_train, Y_train = [], []
    cat_count = {cat: 0 for cat in set(Y_orig)}
    for inst, label in zip(X_orig, Y_orig):
        # If category occured enough, stop adding
        # We do not downsample "other", unless we specified also other
        if cat_count[label] < max_inst or (label == "other" and not also_other):
            X_train.append(inst)
            Y_train.append(label)
            cat_count[label] += 1
    # Check if we have to do something with the "other" category
    if also_other:
        return X_train, Y_train
    # By default, we now downsample the "other" label to be the same size as the full data set
    return down_sample_other(X_train, Y_train)


def feature_count(vectorizer, X_train):
    '''For each feature get its name, the number of docs it appears in and the total amount'''
    count_dict = {}
    # Get the feature matrix
    matrix = vectorizer.fit_transform(X_train)
    # Loop over names and full count
    for name, count in zip(vectorizer.get_feature_names(), matrix.sum(axis=0).tolist()[0]) :
        count_dict[name] = count
    return count_dict


def ngrams_are_words(ngram, word_list):
    '''Check if an ngram actually consists of English words'''
    for word in ngram:
        if word not in word_list:
            return False
    return True


def print_division(label_names, labels):
    '''Print label division of training set'''
    print ("\nLabel division:")
    print(tabulate([[label, labels.count(label)] for label in label_names]))
    print()


def print_best_features(vectorizer, clf, X_train, only_words):
    '''Prints features with the highest coefficient values, per class'''
    # Check if we only want to print features that are English words
    if only_words:
        word_list = load_json_dict(word_list_file)
    # We also want to get the number of docs the feature occurs in (and total amount)
    count_dict = feature_count(vectorizer, X_train)
    # Now get the best features and print them
    num_features = 10
    labels = clf.named_steps['cls'].classes_
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(labels):
        top = np.argsort(clf.named_steps['cls'].coef_[i])
        # Get the best features, order from best to worst
        # Select a bit more because we might filter non-English words later
        sort_top = top[-(num_features)*10:][::-1]
        # Print features most indicative of this class
        print ("\nBest features for " + class_label + ":\n")
        done = []
        for j in sort_top:
            # Stop if we output enough features already
            if len(done) >= num_features:
                break
            if not only_words or ngrams_are_words(feature_names[j].split(), word_list):
                print(feature_names[j], round(clf.named_steps['cls'].coef_[i][j], 2), "({0})".format(round(count_dict[feature_names[j]], 1)))
                done.append(feature_names[j])
        # Command to show words as just a list
        print ("\n" + class_label + ":", ", ".join(done) + "\n")


def filter_by_category(instances, labels, filter_categories):
    '''Filter instances if they belong to a certain category'''
    new_insts, new_labels = [], []
    for inst, label in zip(instances, labels):
        if label not in filter_categories:
            new_insts.append(inst)
            new_labels.append(label)
    return new_insts, new_labels


def get_data(input_file, filter_categories, down_sample, also_other, limit_train):
    '''Read in the specified data. Perhaps filter certain categories or down-sample them'''
    # Read DSI data
    X_train, Y_train = read_dsi_corpus(input_file)

    # Maybe filter instances that have a certain label
    if filter_categories:
        X_train, Y_train = filter_by_category(X_train, Y_train, filter_categories)

    # Train on less data perhaps
    if limit_train > 0:
        if limit_train >= len(X_train):
            print("WARNING: limiting has no effect")
        X_train = X_train[0:limit_train]
        Y_train = Y_train[0:limit_train]

    # Maybe downsample a specific class
    if down_sample > 0:
        X_train, Y_train = down_sample_data(X_train, Y_train, down_sample, also_other)
    return X_train, Y_train


def read_test_data(test_file, test_no_labels, filter_categories):
    '''Read in test data with or without labels and perhaps filter certain categories'''
    Y_test = []
    if test_no_labels:
        # No labels specified, filter them anyway just to be sure and also this
        # allows to use the write to file functionality for labeled files anyway
        X_test = [x.strip().split('\t')[0] for x in open(test_file, 'r')]
    else:
        X_test, Y_test = read_dsi_corpus(test_file)
        # Also potentially filter categories here
        if filter_categories:
            X_test, Y_test = filter_by_category(X_test, Y_test, filter_categories)
    return X_test, Y_test


def main():
    '''Main function'''
    args = create_arg_parser()
    # Nice for reproducibility when running lots of exps
    print ("Generated by:\npython {0}\n".format(" ".join(sys.argv)))

    # Read in training data
    X_train, Y_train = get_data(args.input_file, args.filter_categories, args.down_sample, args.also_other, args.limit_train)

    # Read in dev/test set if we specified it
    if args.test_file:
        X_test, Y_test = read_test_data(args.test_file, args.test_no_labels, args.filter_categories)

    # Convert the texts to vectors
    if args.tfidf:
        vec = TfidfVectorizer(min_df=args.min_df, ngram_range=(1,2))
    else:
        # Simple BoW vectorizer
        vec = CountVectorizer(min_df=args.min_df, ngram_range=(1,2))

    # Choose the algorithm
    if args.algorithm == "nb":
        clf = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    elif args.algorithm == "svm":
        clf = svm.LinearSVC(C=1)
        # Use the CalibratedClassifier so we can use predict_proba later
        if args.probabilities:
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf = Pipeline([('vec', vec), ('cls', clf)])

    # Do we do 1v1 or 1 v rest?
    if args.one_vs_rest:
        clf = Pipeline([('cls', OneVsRestClassifier(clf))])

    # Train & test on separate set
    if args.test_file:
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print_division(clf.named_steps['cls'].classes_, Y_train)
        # If labels, do classification report, else print predictions to specified file
        if args.test_no_labels:
            if args.probabilities:
                # Get the probabilities and not the categories
                probs = clf.predict_proba(X_test)
                # Write as space separated lines with probs
                # Add the headers as well so we remember
                out_lines = [" ".join(clf.named_steps['cls'].classes_)] + [" ".join([str(x) for x in row]) for row in probs]
            else:
                # Write as: sentence TAB pred_label
                out_lines = ["{0}\t{1}".format(sent, lab) for sent, lab in zip(X_test, Y_pred)]
            write_to_file(out_lines, args.test_no_labels)
        else:
            print (classification_report(Y_test, Y_pred, digits=3))
    # Cross validation
    else:
        Y_pred = cross_val_predict(clf, X_train, Y_train, n_jobs=3, cv=args.cross_validate)
        print_division(clf.named_steps['cls'].classes_, Y_train)
        print (classification_report(Y_train, Y_pred, digits=3))

    # Feature analysis, only possible for SVM
    if args.features and args.algorithm == "svm" and not args.one_vs_rest:
        print_best_features(vec, clf, X_train, args.only_word_features)

    # Confusion matrix if we want
    if args.confusion:
        Y_plot = Y_test if args.test_file else Y_train
        plot_confusion_matrix(confusion_matrix(Y_plot, Y_pred), [to_small[c] for c in clf.named_steps['cls'].classes_], args.confusion, normalize=False)


if __name__ == "__main__":
    main()
