#!/usr/bin/env python

from __future__ import division

import re
import csv
import StringIO
import pydot

from sklearn import tree as sklearn_tree
from sklearn import svm as sklearn_svm

from sklearn.ensemble import ExtraTreesClassifier

import numpy as np

import matplotlib.pyplot as plt

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

    feature_names = ['Animated Feature Film', 'Cinematography', 'Costume Design', 'Directing', 
                     'Documentary (Feature)', 'Film Editing', 'Foreign Language Film', 'Makeup', 
                     'Music (Scoring)', 'Music (Song)', 'Sound Editing', 'Visual Effects',
                     'Writing', 'Art Direction',
                     'Actor -- Leading Role', 'Actor -- Supporting Role',
                     'Actress -- Leading Role', 'Actress -- Supporting Role']
    



    #feature_names = ["Cinematography", "Writing", "Directing", "Film Editing"]
    num_features = len(feature_names)
    
    feature_list, classification_list = parse_data(feature_names)

    # Decision Tree
    '''
    tree = sklearn_tree.DecisionTreeClassifier()
    tree = tree.fit(feature_list, classification_list)

    with open("tree.dot", 'w') as f:
        f = sklearn_tree.export_graphviz(tree, out_file=f, 
                                         feature_names=feature_names)

    # Support Vector Machine
    svm = sklearn_svm.SVC()
    svm.fit(feature_list, classification_list)
    '''
    
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  compute_importances=True,
                                  random_state=0)
    
    forest.fit(feature_list, classification_list)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    #indices = np.argsort(importances)[::-1]
    indices = np.argsort(importances)

    feature_titles = [' '.join(feature_names[idx].replace('--','').split()[0:2]) 
                      for idx in indices]
    
    # Print the feature ranking
    print "Feature ranking:"
    
    for f in xrange(num_features):
        print "%d. feature %d (%f)" % (f+1, indices[f], importances[indices[f]])
        
    # Plot the feature importances of the forest
    '''
    plt.figure()
    plt.title("Feature importances")
    plt.bar(xrange(num_features), importances[indices],
           color="r", align="center")
    
    plt.xticks([point - .5 for point in xrange(num_features)], 
               feature_titles, rotation=45, size="small")
    plt.xlim([-1, num_features])
    plt.savefig("ForestFeatures.pdf")
    '''
    
    importance_max = 0.5
    fig = plt.figure(figsize=(9,7))
    ax1 = fig.add_subplot(111)
    #plt.subplots_adjust(left=0.115, right=0.88)
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


if __name__ == "__main__":
    main()
