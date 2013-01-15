#!/usr/bin/env python

import re
import csv
import StringIO
import pydot

from sklearn import tree as sklearn_tree
from sklearn import svm as sklearn_svm

from sklearn.ensemble import ExtraTreesClassifier

import numpy as np

def parse_data(FEATURES):

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

    #FEATURES = MOVIE_FEATURES + ACTOR_FEATURES

    #NUM_FEATURES = len(MOVIE_FEATURES) + len(ACTOR_FEATURES)
    NUM_FEATURES = len(FEATURES)

    # What we are trying to predict
    CATEGORY = ['Best Picture']


    academy_data = csv.reader(open('academy_awards.csv'), skipinitialspace=True)

    #academy_data = open("academy_awards.csv", "r")

    data = {}

    next(academy_data)
    for datum in academy_data:
        year = datum[0]
        category = datum[1].strip()
        
        if category in MOVIE_FEATURES:
            film = datum[2]
            person = datum[3]

        elif category in ACTOR_FEATURES:
            actor = datum[2]
            
            if '; and' in datum[3]:
                continue

            if 'To Charles Chaplin, for acting, writing, directing and producing The Circus.' \
                    in datum[2]: continue

            m = re.match(r"(.*?)\{(.*?)\}", datum[3])
            if m==None:
                raise Exception
            film = m.group(1) #datum[3] # Need regex for: "movie {character}"

        elif category in CATEGORY:
            film = datum[2]
            producer = datum[3]

        else:
            continue

        Won = 0
        if datum[4] == "NO": Won = 1
        elif datum[4] == "YES": Won = 2

        movie_id = (film, year)

        if movie_id not in data:
            data[movie_id] = {}
            
        data[movie_id][category] = Won


    # Now, turn the dictionary into lists of
    # features and classification

    feature_list = []
    classification_list = []
    for movie_id, movie_dict in data.iteritems():
        feature_vector = [0 for feature in range(NUM_FEATURES)]
        for feature, value in movie_dict.iteritems():
            if feature not in FEATURES: continue
            feature_vector[FEATURES.index(feature)] = value
        feature_list.append(feature_vector)
        classification_list.append(movie_dict.get("Best Picture", 0))

    return (feature_list, classification_list)


def main():

    feature_names = ["Cinematography", "Writing", "Directing", "Film Editing"]
    num_features = len(feature_names)
    
    feature_list, classification_list = parse_data(feature_names)

    #sample_mask = [False for feature in feature_names]    
    #for feature in FEATURES_TO_USE:
    #    sample_mask[feature_names.index(feature)] = True

    #for features, classification in zip(feature_list, classification_list):
    #    print features, classification


    # Decision Tree
    tree = sklearn_tree.DecisionTreeClassifier()
    tree = tree.fit(feature_list, classification_list)

    with open("tree.dot", 'w') as f:
        f = sklearn_tree.export_graphviz(tree, out_file=f, 
                                         feature_names=feature_names)

    # Support Vector Machine
    svm = sklearn_svm.SVC()
    svm.fit(feature_list, classification_list)

    
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  compute_importances=True,
                                  random_state=0)
    
    forest.fit(feature_list, classification_list)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print "Feature ranking:"
    
    for f in xrange(num_features):
        print "%d. feature %d (%f)" % (f+1, indices[f], importances[indices[f]])
        
    # Plot the feature importances of the forest
    import pylab as pl
    pl.figure()
    pl.title("Feature importances")

    #pl.bar(xrange(num_features), importances[indices],
    #       color="r", yerr=std[indices], align="center")
    pl.bar(xrange(num_features), importances[indices],
           color="r", align="center")
    
    pl.xticks(xrange(num_features), indices)
    pl.xlim([-1, num_features])
    #pl.show()
    pl.savefig("ForestFeatures.pdf")


if __name__ == "__main__":
    main()
