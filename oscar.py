#!/usr/bin/env python

from __future__ import division

import re
import csv
import StringIO
import pydot
import copy

from sklearn import tree as sklearn_tree
from sklearn import svm as sklearn_svm

from sklearn.ensemble import ExtraTreesClassifier

import numpy as np

import matplotlib.pyplot as plt

def parse_data(FEATURES, nomination_data_only=True):

    # Feature Key:
    # 0 - Neither Nominated Nore Won
    # 1 - Nominated but didn't win
    # 2 - Nominated and won

    # What features we're interested in
    MOVIE_FEATURES = ['Animated Feature Film', 'Cinematography', 'Costume Design', 'Directing', 
                      'Documentary (Feature)', 'Film Editing', 'Foreign Language Film', 'Makeup', 
                      'Music (Scoring)', 'Music (Song)', 'Sound Editing', 'Visual Effects',
                      'Writing', 'Art Direction']
    
    ACTOR_FEATURES = ['Actor -- Leading Role', 'Actor -- Supporting Role',
                      'Actress -- Leading Role', 'Actress -- Supporting Role']

    #NUM_FEATURES = len(MOVIE_FEATURES) + len(ACTOR_FEATURES)
    NUM_FEATURES = len(FEATURES)

    # What we are trying to predict
    CLASSIFICATION = ['Best Picture']

    academy_data = csv.reader(open('academy_awards.csv'), skipinitialspace=True)

    data = {}

    next(academy_data)

    for datum in academy_data:

        # Skip based on a few exceptions
        if '; and' in datum[3]:
            continue
        if 'To Charles Chaplin, for acting, writing, directing and producing The Circus.' \
                in datum[2]: continue

        datum = [item.strip() for item in datum]

        (year, category) = datum[0:2]
        
        if category in MOVIE_FEATURES:
            (film, person, won) = datum[2:5]

        elif category in ACTOR_FEATURES:
            (actor, film_str, won) = datum[2:5]
            m = re.match(r"(.*?)\{(.*?)\}", film_str)
            if m==None:
                raise Exception
            film = m.group(1).strip()

        elif category in CLASSIFICATION:
            (film, producer) = datum[2:4]
            producer = datum[3]

        else:
            continue

        # All movies get a '0' for all features by default
        # If they're only nominated, they get a 1 for that feature.
        # If they win, they get a 2 for that feature.
        Won = 0
        if datum[4] == "NO": Won = 1
        elif datum[4] == "YES": Won = 2
        else:
            raise Exception

        # If we're only using 'nominated' as a feature category
        # then we ignore if it won a feature category and only
        # classify it based on nomination
        if nomination_data_only and category not in CLASSIFICATION:
            Won=1

        movie_id = (film, year)

        if movie_id not in data:
            data[movie_id] = {}
            
        data[movie_id][category] = Won

    # Now, turn the dictionary into lists of
    # features and classification
    # Only include movies that got a best
    # picture nomination
    feature_list = []
    classification_list = []
    for movie_id, movie_dict in data.iteritems():
        if "Best Picture" not in movie_dict:
            continue
        feature_vector = [0 for feature in range(NUM_FEATURES)]
        for feature, value in movie_dict.iteritems():
            if feature not in FEATURES: continue
            feature_vector[FEATURES.index(feature)] = value
        feature_list.append(feature_vector)
        classification_list.append(movie_dict.get("Best Picture", 0))

    return (feature_list, classification_list)


def test_classification(classifier, features, classifications, **kwargs):
    """ Assume that a classifier has a 'fit' method
    """

    print "Classificatin accuracy for: ", classifier.__class__.__name__

    classifier = copy.deepcopy(classifier)

    training_features, training_class = features[1::2], classifications[1::2]
    validation_features, validation_class = features[0::2], classifications[0::2]

    clf = classifier.fit(training_features, training_class, **kwargs)

    correct = 0
    incorrect = 0

    for features, classification in zip(validation_features, validation_class):
        prediction = clf.predict(features)[0]
        if prediction == classification: correct += 1
        else: incorrect += 1
        
    accuracy = correct / (correct + incorrect)

    print "Num Correct: %s" % correct
    print "Num Incorrect: %s" % incorrect
    print "Accuracy: %s" % accuracy


def main():

    feature_names = ['Animated Feature Film', 'Cinematography', 'Costume Design', 'Directing', 
                     'Documentary (Feature)', 'Film Editing', 'Foreign Language Film', 'Makeup', 
                     'Music (Scoring)', 'Music (Song)', 'Sound Editing', 'Visual Effects',
                     'Writing', 'Art Direction',
                     'Actor -- Leading Role', 'Actor -- Supporting Role',
                     'Actress -- Leading Role', 'Actress -- Supporting Role']

    features_minimal = ['Cinematography', 'Directing', 
                        'Film Editing', 'Writing']

    #feature_names = ["Cinematography", "Writing", "Directing", "Film Editing"]
    num_features = len(feature_names)

    # For trees, use the minimal features
    feature_list, classification_list = parse_data(features_minimal, True)

    # Decision Tree
    tree = sklearn_tree.DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=4)

    test_classification(tree, feature_list, classification_list)
    tree = tree.fit(feature_list, classification_list)
    
    with open("tree.dot", 'w') as f:
        f = sklearn_tree.export_graphviz(tree, out_file=f, 
                                         feature_names=features_minimal)

    dot_data = StringIO.StringIO()
    sklearn_tree.export_graphviz(tree, out_file=dot_data, 
                                 feature_names=features_minimal)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    #  {'dot': '', 'twopi': '', 'neato': '', 'circo': '', 'fdp': ''}
    graph.write_pdf("tree.pdf", prog='dot') 

    # Use the full feature list for SVM and Random Forests
    feature_list, classification_list = parse_data(feature_names, True)

    # Support Vector Machine
    svm = sklearn_svm.SVC()
    test_classification(svm, feature_list, classification_list)
    svm.fit(feature_list, classification_list)
    
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  compute_importances=True,
                                  random_state=0)
    test_classification(forest, feature_list, classification_list)
    forest.fit(feature_list, classification_list)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)

    feature_titles = [' '.join(feature_names[idx].replace('--','').split()[0:2]) 
                      for idx in indices]
    
    # Print the feature ranking
    importance_max = 0.5
    fig = plt.figure(figsize=(9,7))
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(left=0.25, right=0.88)
    fig.canvas.set_window_title('Oscars')
    pos = np.arange(num_features)+0.5    #Center bars on the Y-axis ticks
    rects = ax1.barh(pos, importances[indices], 
                     align='center', height=0.5, color='m')

    ax1.axis([0, importance_max, 0, num_features])
    plt.yticks(pos, feature_titles)
    ax1.set_title('Oscars')
    #plt.text(.25, 2.0, 'Importance', horizontalalignment='center')
    plt.xlabel("Relative Importance")
    plt.savefig("ForestFeatures.pdf")


    # Do some plotting
    feature_list, classification_list = parse_data(feature_names, False)

    class feature_stat:

        feature = {'None' : 0, 'Nominated' : 1, 'Won': 2}

        def __init__(self, name):
            self.name = name
            self.array = [ [0,0,0],  # Not Nominated BP
                           [0,0,0],  # Nominated BP
                           [0,0,0] ] # Won BP

        def update(self, features, classification):
            # Get the feature
            idx = feature_names.index(self.name)
            if idx not in range(len(feature_names)):
                raise Exception
            feature_value = features[idx]
            is_best_picture = classification
            self._update(feature_value, is_best_picture)

        def _update(self, feature_value, is_best_picture):
            assert feature_value in [0, 1, 2]
            assert is_best_picture in [0, 1, 2]
            self.array[is_best_picture][feature_value] += 1

        def _frac(self, feature_value, is_best_picture):
            assert feature_value in [0, 1, 2]
            assert is_best_picture in [0, 1, 2]
            return self.array[is_best_picture][feature_value]/sum(self.array[is_best_picture])

        def frac(self, feature_string, best_picture_string):
            return self._frac(self.feature[feature_string], 
                              self.feature[best_picture_string])

        def percentage(self, feature_string):
            return "%.1f%% / %.1f%%" % (self.frac(feature_string, "Won")*100,
                                        self.frac(feature_string, "Nominated")*100)

        def row(self):
            return [self.percentage("Won"), 
                    self.percentage("Nominated"), 
                    self.percentage("None")] 

    best_actor = feature_stat('Actor -- Leading Role')
    best_actress = feature_stat('Actress -- Leading Role')
    best_supp_actor = feature_stat('Actor -- Supporting Role')
    best_supp_actress = feature_stat('Actress -- Supporting Role')

    for features, classification in zip(feature_list, classification_list):
        best_actor.update(features, classification)
        best_actress.update(features, classification)
        best_supp_actor.update(features, classification)
        best_supp_actress.update(features, classification)

    table = []

    table.append(["Best Actor"] + best_actor.row())
    table.append(["Best Actress"] + best_actress.row())
    table.append(["Best Supporting Actor"] + best_supp_actor.row())
    table.append(["Best Supporintg Actress"] + best_supp_actress.row())

    from prettytable import PrettyTable
    pretty_table = PrettyTable(["Category", "Won Award", "Nominated", "Not Nominated"])
    #pretty_table.set_field_align("Category", "l")
    for row in table:
        pretty_table.add_row(row)

    print '\n'
    print pretty_table
    print pretty_table.get_html_string()


if __name__ == "__main__":
    main()
