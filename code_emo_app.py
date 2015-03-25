# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:49:44 2015

@author: charlesmartens
"""

from pattern import en
from pattern.en import conjugate, lemma, lexeme
#from pattern.en import tenses, PAST, PL
from pattern.en import wordnet as wn
import enchant  #why not working??
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import pickle
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns

#import flask
#
## Initialize the app
#app = flask.Flask(__name__)

# Homepage
#@app.route("/")
#def viz_page():
#    """ Homepage: serve our visualization page, my_webpage.html
#    """
#    with open("my_webpage.html", 'r') as viz_file:
#        return viz_file.read()


#SEE END OF FILE FOR KEYF FUNCTION


######################################################################
#Use this dict for now as the emotion dictionary. tweak later.
#add words from list to add below. and take out lots of words
#whose primary meaning is non-emo.
#to retrieve pickle:
with open('clore_and_storm_words_Mar19_dict.pkl', 'r') as picklefile:
    clore_and_storm_Mar19_dict = pickle.load(picklefile)
######################################################################


################################################################
#to get waking reports corpus:
with open('waking_corpus_clean.pkl', 'r') as picklefile:
    waking_corpus_clean_2 = pickle.load(picklefile)
################################################################


################################################################
#to get dream reports corpus:
with open('dream_corpus_clean.pkl', 'r') as picklefile:
    dream_corpus_clean_2 = pickle.load(picklefile)
################################################################


#in each set of text docs
#change all words in reports to lowercase
def corpus_lowercase(corpus):
    corpus_lower =[]
    for report in corpus:
        textblob_report = TextBlob(report)
        new_report = ' '.join([word.lower() for word in textblob_report.words]) 
        corpus_lower.append(new_report) 
    return corpus_lower


#corret spelling in reports
def corpus_spelling_correct(corpus):
    corpus_spell_correct =[]
    for report in corpus:
        textblob_report = TextBlob(report)
        report_spelled = textblob_report.correct()
        corpus_spell_correct.append(report_spelled)
    return corpus_spell_correct
    
    
#replace all emotion words in report corpus with the root word. 
#i.e, replace scare and scary with scared. 
def replace_emo_words_w_root(corpus, emotion_to_root_dict):
    corpus_replaced_emotions = []
    for report in corpus:   
        for key in emotion_to_root_dict.keys():
            for word in emotion_to_root_dict[key]:
                report = report.replace(word, key)
        corpus_replaced_emotions.append(report)
    return corpus_replaced_emotions


#create dict where emotion_complete category is the key and the values are whether absent or present in each report
def count_docs_w_ea_emotion(corpus, emotion_to_root_dict):
    count_of_ea_emotion_dict = defaultdict(list)
    for report in corpus:   
        for emotion in emotion_to_root_dict.keys():
            if emotion in report:
                count_of_ea_emotion_dict[emotion].append(1)
            else:
                count_of_ea_emotion_dict[emotion].append(0)
    return count_of_ea_emotion_dict


#sort emotions words alphabetically 
def sort_emotion_counts_alphabetically(emotion_to_count_dict):
    """Takes dictionary with each emotion and how many docs it appears in (from a corpus) and sorts the emotions
    (and corresponding counts) from a to z"""
    words_to_counts_list = []
    for key, value in emotion_to_count_dict.iteritems():
        words_to_counts_list.append([key, sum(value)])
    def get_key(item):
        return item[0]
    sorted_emotions_words_to_counts = sorted(words_to_counts_list, key=get_key)
    return sorted_emotions_words_to_counts


#compute ratio of emotions in dream reports over waking reports
def get_emotion_corpus1_to_corpus2_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2):
    """Takes list of emotions and their counts sorted alphabetically and computes emotion ratios.
    Then sorts these emotion ratios from highest to lowest"""
    emotions_ratio_list = [] 
    for i in range(len(alphabetical_emotion_counts_corpus1)):
        emotion = alphabetical_emotion_counts_corpus1[i][0]
        ratio = float((alphabetical_emotion_counts_corpus1[i][1] + 10)) / float((alphabetical_emotion_counts_corpus2[i][1] + 10))
        emotions_ratio_list.append([emotion, ratio])
    def get_key(item):
        return item[1]    
    sorted_emotion_corpus1_to_corpus2_ratios = sorted(emotions_ratio_list, key=get_key, reverse=True)  #put () after get key???
    return sorted_emotion_corpus1_to_corpus2_ratios


#compute ratio of emotions in waking reports over dream reports
def get_emotion_corpus2_to_corpus1_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2):
    """Takes list of emotions and their counts sorted alphabetically and computes emotion ratios.
    Then sorts these emotion ratios from highest to lowest"""
    emotions_ratio_list = [] 
    for i in range(len(alphabetical_emotion_counts_corpus1)):
        emotion = alphabetical_emotion_counts_corpus1[i][0]
        ratio = float((alphabetical_emotion_counts_corpus2[i][1] + 10)) / float((alphabetical_emotion_counts_corpus1[i][1] + 10))
        emotions_ratio_list.append([emotion, ratio])
    def get_key(item):
        return item[1]    
    sorted_emotion_corpus2_to_corpus1_ratios = sorted(emotions_ratio_list, key=get_key, reverse=True)
    return sorted_emotion_corpus2_to_corpus1_ratios


#plot  -  TURN INTO FUNCTION:
def plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios):
    X = [word[0] for word in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
    Y = [freq[1] for freq in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
    fig = plt.figure(figsize=(15, 5))  #add this to set resolution: , dpi=100
    sns.barplot(x = np.array(range(len(X))), y = np.array(Y))
    sns.despine(left=True)
    plt.title('Emotion-words Most Representative of ' + corpus1_name, fontsize=17)
    plt.xticks(rotation=75)
    plt.xticks(np.array(range(len(X))), np.array(X), rotation=75, fontsize=15)
    plt.ylim(1, 3.05)
    plt.ylabel('Frequency in {} relative to {}'.format(corpus1_name, corpus2_name), fontsize=15)
    #this tight_layout method fixes problem of x-axis labels cut off in saved figure:    
    plt.tight_layout()
    plt.savefig('static_test/corpus1_to_corpus2.png')


#plot  -  TURN INTO FUNCTION:
def plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, sorted_emotion_corpus2_to_corpus1_ratios):
    X = [word[0] for word in sorted_emotion_corpus2_to_corpus1_ratios[:25]]
    Y = [freq[1] for freq in sorted_emotion_corpus2_to_corpus1_ratios[:25]]
    fig = plt.figure(figsize=(15, 5))  #add this to set resolution: , dpi=100
    sns.barplot(x = np.array(range(len(X))), y = np.array(Y))
    sns.despine(left=True)
    plt.title('Emotion-words Most Representative of ' + corpus2_name, fontsize=17)
    plt.xticks(rotation=75)
    plt.xticks(np.array(range(len(X))), np.array(X), rotation=75, fontsize=15)
    plt.ylim(1, 3.05)
    plt.ylabel('Frequency in {} relative to {}'.format(corpus2_name, corpus1_name), fontsize=15)
    plt.tight_layout()
    plt.savefig('static_test/corpus2_to_corpus1.png')


#do for ea corpus:
#corpus_lowercase(corpus)
#corpus_spelling_correct(corpus)
#replace_emo_words_w_root(corpus, emotion_to_root_dict)
#count_docs_w_ea_emotion(corpus, emotion_to_root_dict)
#sort_emotion_counts_alphabetically(emotion_to_count_dict)

def corpus_to_alphabetical_emotion_counts(corpus, emotion_to_root_dict):
    corpus_lower = corpus_lowercase(corpus)
    #corpus_lower_spelling = corpus_spelling_correct(corpus_lower)
    corpus_simplify_emo_words = replace_emo_words_w_root(corpus_lower, emotion_to_root_dict)
    emotion_to_count_dict = count_docs_w_ea_emotion(corpus_simplify_emo_words, emotion_to_root_dict)
    alphabetical_emotions_w_counts_list = sort_emotion_counts_alphabetically(emotion_to_count_dict)
    return alphabetical_emotions_w_counts_list


#corpus_lower = corpus_lowercase(waking_corpus_clean_2)
#len(corpus_lower)
#corpus_lower_spelling = corpus_spelling_correct(corpus_lower)
#len(corpus_lower_spelling)  #this took forever to compute!!! elim from function for now.
#corpus_simplify_emo_words = replace_emo_words_w_root(corpus_lower, clore_and_storm_Mar19_dict)
#len(corpus_simplify_emo_words)
#emotion_to_count_dict = count_docs_w_ea_emotion(corpus_simplify_emo_words, clore_and_storm_Mar19_dict)
#len(emotion_to_count_dict)
#alphabetical_emotions_w_counts_list = sort_emotion_counts_alphabetically(emotion_to_count_dict)
#len(alphabetical_emotions_w_counts_list)
#alphabetical_emotions_w_counts_list[:15]


#then get ratios and plot:
#get_emotion_corpus1_to_corpus2_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2)
#get_emotion_corpus2_to_corpus1_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2)
#plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios)
#plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, sorted_emotion_corpus2_to_corpus1_ratios)

def plot_alphabetical_lists(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2, corpus1_name, corpus2_name):
    corpus1_to_corpus2_ratios = get_emotion_corpus1_to_corpus2_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2)
    corpus2_to_corpus1_ratios = get_emotion_corpus2_to_corpus1_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2)
    plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, corpus1_to_corpus2_ratios)
    plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, corpus2_to_corpus1_ratios)


##############################################################################
#master function -- takes input of corpuses and outputs 2 plots:
def corpuses_to_plot(corpus1, corpus2, corpus1_name, corpus2_name, emotion_to_root_dict):
    corpus1_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(corpus1, emotion_to_root_dict)
    corpus2_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(corpus2, emotion_to_root_dict)
    plot_alphabetical_lists(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list, corpus1_name, corpus2_name)
###############################################################################

#corpus1_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(dream_corpus_clean_2, clore_and_storm_Mar19_dict)
#corpus1_alphabetical_counts_list[:20]
#corpus2_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(waking_corpus_clean_2, clore_and_storm_Mar19_dict)
#corpus2_alphabetical_counts_list[:20]
#corpus1_to_corpus2_ratios = get_emotion_corpus1_to_corpus2_ratios(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list)
#corpus1_to_corpus2_ratios[:20]  #this isn't sorted by number. should be i think
#plot_alphabetical_lists(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list, 'Dreams', 'Real-life')

#run master function. takes each corpus of docs and the name for each corpus, and the dictionary of root emotion word to variations of that word
#corpuses_to_plot(dream_corpus_clean_2, waking_corpus_clean_2, 'Dreams', 'Real-life', clore_and_storm_Mar19_dict)

#get input function here to input two corpuses



# Get corpus and plot two graphs
#@app.route("/graphs", methods=["POST"])
#def plot_graphs():
#    """  
#    """

#this worked for showing x-labes on saved fig:
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

corpuses_to_plot(dream_corpus_clean_2, waking_corpus_clean_2, 'Dreams', 'Real-life', clore_and_storm_Mar19_dict)


#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
#app.run(host='0.0.0.0', port=80)












