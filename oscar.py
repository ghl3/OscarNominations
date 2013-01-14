#!/usr/bin/env python

import re
import csv

def main():

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

    # What we are trying to predict
    CATEGORY = ['Best Picture']


    academy_data = csv.reader(open('academy_awards.csv'), skipinitialspace=True)

    #academy_data = open("academy_awards.csv", "r")

    data = {}

    next(academy_data)
    for datum in academy_data:
        #datum = award.replace('"', '').replace("'", '').split(',')
        #datum = award.split(',')
        year = datum[0]
        category = datum[1].strip()
        
        if category in MOVIE_FEATURES:
            film = datum[2]
            person = datum[3]

        elif category in ACTOR_FEATURES:
            actor = datum[2]
            
            if '; and' in datum[3]:
                continue

            if 'To Charles Chaplin, for acting, writing, directing and producing The Circus.' in datum[2]: continue

            print datum[3],
            m = re.match(r"(.*?)\{(.*?)\}", datum[3])
            if m==None:
                print datum
                raise Exception
            else:
                print m, m.group()
            film = m.group(1) #datum[3] # Need regex for: "movie {character}"

        else:
            print "Unknown category: ", category
            continue

        Won = 0
        if datum[4] == "NO": Won = 1
        elif datum[4] == "YES": Won = 2

        movie_id = (film, year)

        if movie_id not in data:
            data[movie_id] = {}
            
        data[movie_id][category] = Won


    print data


if __name__ == "__main__":
    main()
