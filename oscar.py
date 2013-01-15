#!/usr/bin/env python

import re
import csv
import StringIO

from sklearn import tree

def parse_data():

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

    FEATURES = MOVIE_FEATURES + ACTOR_FEATURES

    NUM_FEATURES = len(MOVIE_FEATURES) + len(ACTOR_FEATURES)

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

    feature_list, classification_list = parse_data()

    for features, classification in zip(feature_list, classification_list):
        print features, classification
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(feature_list, classification_list)

    with open("tree.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

if __name__ == "__main__":
    main()
