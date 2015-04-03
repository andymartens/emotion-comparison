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
from matplotlib import rcParams

pd
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


#corret spelling in reports  - this was ver slow so cut out of final function
#faster way to get to work? incorp in previous f so don't have to textblob it twice?
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


##
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


#plot  -  COMMENT THIS OUT - NOW REPLACED WITH OTHER FUNCTION:
#def plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios):
#    X = [word[0] for word in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
#    Y = [freq[1] for freq in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
#    fig = plt.figure(figsize=(10, 6))  #add this to set resolution: , dpi=100
#    sns.barplot(x = np.array(range(len(X))), y = np.array(Y))
#    sns.despine(left=True)
#    plt.title('Emotion-words Most Representative of ' + corpus1_name, fontsize=17)
#    plt.xticks(rotation=75)
#    plt.xticks(np.array(range(len(X))), np.array(X), rotation=75, fontsize=15)
#    plt.ylim(1, 3.05)
#    plt.ylabel('Frequency in {} relative to {}'.format(corpus1_name, corpus2_name), fontsize=15)
#    #this tight_layout method fixes problem of x-axis labels cut off in saved figure:    
#    plt.tight_layout()
#    plt.savefig('static_test/corpus1_to_corpus2.png')

# alt version of above function, to play with:
# this is working ok. need to reverse order so biggest vaules on top.
# def plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios):
#X = [word[0] for word in sorted_emotion_corpus1_to_corpus2_ratios[:20]]
#Y = [freq[1] for freq in sorted_emotion_corpus1_to_corpus2_ratios[:20]]
#plt.figure(figsize=(8, 12))  #add this to set resolution: , dpi=100
#barlist = plt.barh(np.array(range(len(X))), np.array(Y))
#sns.despine(left=True)
#plt.title('Emotion-words Most Representative of ' + corpus1_name, fontsize=17)
##plt.xticks(rotation=75)
#plt.yticks(np.array(range(len(X))) + .5, np.array(X), fontsize=15)
#plt.xlim(1, 3.05)
#plt.xlabel('Frequency in {} relative to {}'.format(corpus1_name, corpus2_name), fontsize=15)
##this tight_layout method fixes problem of x-axis labels cut off in saved figure:    
#for i in range(len(X)):
#    if Y[i] < 1.6:    
#        barlist[i].set_color((0.0, 0.6, 0.0))
#        barlist[i].set_alpha(Y[i]/3)
#    else:    
#        barlist[i].set_color((0.6, 0.0, 0.0))
#        barlist[i].set_alpha(Y[i]/3)    
#plt.tight_layout()
#plt.savefig('static_test/corpus1_to_corpus2.png')
#

# get Xs and Ys to play w:
corpus1_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(dream_corpus_clean_2, root_to_variations_dict)
corpus2_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(waking_corpus_clean_2, root_to_variations_dict)
corpus1_to_corpus2_ratios = get_emotion_corpus1_to_corpus2_ratios(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list)
corpus2_to_corpus1_ratios = get_emotion_corpus2_to_corpus1_ratios(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list)
sorted_emotion_corpus1_to_corpus2_ratios = corpus1_to_corpus2_ratios
sorted_emotion_corpus2_to_corpus1_ratios = corpus2_to_corpus1_ratios
corpus1_name = 'Dreams'
corpus2_name = 'Real Life'

# above works ok. but compare to this syntax from web:


def plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios, corresponding_root_to_ratings_dict):
    #set some graph parameters:
    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 150
    #rcParams['axes.color_cycle'] = dark2_colors
    rcParams['lines.linewidth'] = 2
    rcParams['axes.facecolor'] = 'white'  #this is the background color of the grid area
    rcParams['font.size'] = 14
    rcParams['patch.edgecolor'] = 'white'
    #rcParams['patch.facecolor'] = dark2_colors[0]
    rcParams['font.family'] = 'StixGeneral'

    #a function that gets called later in this function
    def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
        """
        Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
        The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
        """
        ax = axes or plt.gca()
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)
        ax.spines['left'].set_visible(left)
        ax.spines['bottom'].set_visible(bottom)
        #turn off all ticks
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        #now re-enable visibles
        if top:
            ax.xaxis.tick_top()
        if bottom:
            ax.xaxis.tick_bottom()
        if left:
            ax.yaxis.tick_left()
        if right:
            ax.yaxis.tick_right()

    emotion = [word[0] for word in sorted_emotion_corpus1_to_corpus2_ratios[:20]]
    ratio = [round(freq[1], 2) for freq in sorted_emotion_corpus1_to_corpus2_ratios[:20]]
    plt.figure(figsize=(5, 8))
    pos = np.arange(len(emotion))
    plt.title(corpus1_name + ' / ' + corpus2_name, horizontalalignment='center')
    barlist = plt.barh(pos, ratio)  # the plt.plot function returns certain info, e.g.,
    # the values for bars or lines or whatever is plotted. and these values are now put
    # into barlist object so i can use them below, e.g., loop through them and give the
    # bars some new attributes

    #add the numbers to the side of each bar
    for p, e, r in zip(pos, emotion, ratio):
        plt.annotate(str(r), xy=(r + .25, p + .5), va='center')

    #shade the bars based on intensity. it's looping through these values for the bars
    # and re-draws or re-does the bar with the new attribute
    for i in range(len(emotion)):
        the_emotion = emotion[i]
        valence = corresponding_root_to_ratings_dict[the_emotion][0]
        arousal = corresponding_root_to_ratings_dict[the_emotion][1]
        intensity_1 = corresponding_root_to_ratings_dict[the_emotion][2]
        intensity_2 = (np.abs(valence)/4) * (arousal*10)
        intensity_3 = arousal * np.sqrt(np.abs(valence/4))   #hmmm, this isn't bad. good for now.
        if corresponding_root_to_ratings_dict[the_emotion][0] > 0:
            barlist[i].set_color((0.0, 0.85, 0.0))
            barlist[i].set_alpha(intensity_3 / .7)
            print intensity_3
        else:
            barlist[i].set_color((0.99, 0.0, 0.0))
            barlist[i].set_alpha(intensity_3 / .7)
            print intensity_3
    #above, i like using arousal. might not be as good for
    #pos words in the opposite graph, though?

    #cutomize ticks
    ticks = plt.yticks(pos + .5, emotion, fontsize=18)
    xt = plt.xticks()[0]
    plt.xticks(xt, [' '] * len(xt))
    #minimize chartjunk
    remove_border(left=False, bottom=False)
    plt.grid(axis = 'x', color ='white', linestyle='-')
    #set plot limits
    plt.ylim(pos.max() + 1, pos.min() - 1)
    plt.xlim(0, 4)
    plt.tight_layout()  #this keeps words from getting cut off
    plt.savefig('static_test/corpus1_to_corpus2.png')  #will need to change this so saves to static folder


plot_ratios_corpus1_to_corpus2('Dreams', 'Real Life', sorted_emotion_corpus1_to_corpus2_ratios, corresponding_root_to_ratings_dict)

############################################################
#intensity_list = [ratings[2] for ratings in corresponding_root_to_ratings_dict.values()]
#np.nanmean(intensity_list)
#np.nanmin(intensity_list)
#np.nanmax(intensity_list)
#
#arousal_list = [ratings[1] for ratings in corresponding_root_to_ratings_dict.values()]
#np.nanmean(arousal_list)
#np.nanmin(arousal_list)
#np.nanmax(arousal_list)
#
#valence_list = [ratings[0] for ratings in corresponding_root_to_ratings_dict.values()]
#np.nanmean(valence_list)
#np.nanmin(valence_list)
#np.nanmax(valence_list)
#


#plot  -  TURN INTO FUNCTION:
#def plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, sorted_emotion_corpus2_to_corpus1_ratios):
#    X = [word[0] for word in sorted_emotion_corpus2_to_corpus1_ratios[:25]]
#    Y = [freq[1] for freq in sorted_emotion_corpus2_to_corpus1_ratios[:25]]
#    fig = plt.figure(figsize=(15, 5))  #add this to set resolution: , dpi=100
#    sns.barplot(x = np.array(range(len(X))), y = np.array(Y))
#    sns.despine(left=True)
#    plt.title('Emotion-words Most Representative of ' + corpus2_name, fontsize=17)
#    plt.xticks(rotation=75)
#    plt.xticks(np.array(range(len(X))), np.array(X), rotation=75, fontsize=15)
#    plt.ylim(1, 3.05)
#    plt.ylabel('Frequency in {} relative to {}'.format(corpus2_name, corpus1_name), fontsize=15)
#    plt.tight_layout()
#    plt.savefig('static_test/corpus2_to_corpus1.png')


#def plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, sorted_emotion_corpus2_to_corpus1_ratios, corresponding_root_to_ratings_dict):
#    #set some graph parameters:
#    rcParams['figure.figsize'] = (10, 6)
#    rcParams['figure.dpi'] = 150
#    #rcParams['axes.color_cycle'] = dark2_colors
#    rcParams['lines.linewidth'] = 2
#    rcParams['axes.facecolor'] = 'white'  #this is the background color of the grid area
#    rcParams['font.size'] = 14
#    rcParams['patch.edgecolor'] = 'white'
#    #rcParams['patch.facecolor'] = dark2_colors[0]
#    rcParams['font.family'] = 'StixGeneral'
#
#    #a function that gets called later in this function
#    def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
#        """
#        Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
#        The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
#        """
#        ax = axes or plt.gca()
#        ax.spines['top'].set_visible(top)
#        ax.spines['right'].set_visible(right)
#        ax.spines['left'].set_visible(left)
#        ax.spines['bottom'].set_visible(bottom)
#        #turn off all ticks
#        ax.yaxis.set_ticks_position('none')
#        ax.xaxis.set_ticks_position('none')
#        #now re-enable visibles
#        if top:
#            ax.xaxis.tick_top()
#        if bottom:
#            ax.xaxis.tick_bottom()
#        if left:
#            ax.yaxis.tick_left()
#        if right:
#            ax.yaxis.tick_right()
#
#
#    emotion = [word[0] for word in sorted_emotion_corpus2_to_corpus1_ratios[:20]]
#    ratio = [round(freq[1], 2) for freq in sorted_emotion_corpus2_to_corpus1_ratios[:20]]
#    #intensity = [.2, .7, .7, .9, .1, .8, .7, .6, .1, .5, 
#    #             .4, .1, .6, .9, .4, .7, .9, .8, .7, .6]
#
#    grad = pd.DataFrame({'ratio' : ratio, 'emotion': emotion, 'intensity': intensity})
#    plt.figure(figsize=(5, 8))
#    #change = grad.change[grad.change > 0]  #in future step, just create one long list of ratios and select the top 20 here (and bottom 20 when do other/second graph)
#    #city = grad.city[grad.change > 0]
#    #intensity = grad.intensity[grad.change > 0]
#    #ratio = grad.ratio
#    #emotion = grad.emotion
#    #intensity = grad.intensity
#
#    pos = np.arange(len(emotion))
#    plt.title(corpus2_name + ' / ' + corpus1_name, horizontalalignment='center')
#    barlist = plt.barh(pos, ratio)  # the plt.plot function returns certain info, e.g.,
#    # the values for bars or lines or whatever is plotted. and these values are now put 
#    # into barlist object so i can use them below, e.g., loop through them and give the 
#    # bars some new attributes
#
#    #add the numbers to the side of each bar
#    for p, e, r in zip(pos, emotion, ratio):
#        plt.annotate(str(r), xy=(r + .25, p + .5), va='center')
#
#    #shade the bars based on intensity. it's looping through these values for the bars
#    # and redrawsor re-does the bar with the new attribute
#    for i in range(len(emotion)):
#        the_emotion = emotion[i]
#        valence = corresponding_root_to_ratings_dict[the_emotion][0]
#        arousal = corresponding_root_to_ratings_dict[the_emotion][1]
#        intensity_1 = corresponding_root_to_ratings_dict[the_emotion][2]
#        intensity_2 = (np.abs(valence)/4) * (arousal*10)
#        intensity_3 = arousal * np.sqrt(np.abs(valence/4))   #hmmm, this isn't bad. good for now.
#        if corresponding_root_to_ratings_dict[the_emotion][0] > 0:    
#            barlist[i].set_color((0.0, 0.85, 0.0))
#            barlist[i].set_alpha(intensity_3 / .7)
#            print intensity_3
#        else:    
#            barlist[i].set_color((0.99, 0.0, 0.0))
#            barlist[i].set_alpha(intensity_3 / .7)
#            print intensity_3
#
#    #above, i like using arousal. might not be as good for
#    #pos words in the opposite graph, though? 
#
#    #cutomize ticks
#    ticks = plt.yticks(pos + .5, emotion)
#    xt = plt.xticks()[0]
#    plt.xticks(xt, [' '] * len(xt))
#    #minimize chartjunk
#    remove_border(left=False, bottom=False)
#    plt.grid(axis = 'x', color ='white', linestyle='-')
#    #set plot limits
#    plt.ylim(pos.max() + 1, pos.min() - 1)
#    plt.xlim(0, 3.5)
#    
#    plt.tight_layout()  #this keeps words from getting cut off
#    plt.savefig('static_test/corpus2_to_corpus1_ALT.png')  #will need to change this so saves to static folder


def plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, sorted_emotion_corpus2_to_corpus1_ratios, corresponding_root_to_ratings_dict):
    #set some graph parameters:
    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 150
    #rcParams['axes.color_cycle'] = dark2_colors
    rcParams['lines.linewidth'] = 2
    rcParams['axes.facecolor'] = 'white'  #this is the background color of the grid area
    rcParams['font.size'] = 14
    rcParams['patch.edgecolor'] = 'white'
    #rcParams['patch.facecolor'] = dark2_colors[0]
    rcParams['font.family'] = 'StixGeneral'

    #a function that gets called later in this function
    def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
        """
        Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
        The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
        """
        ax = axes or plt.gca()
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)
        ax.spines['left'].set_visible(left)
        ax.spines['bottom'].set_visible(bottom)
        #turn off all ticks
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        #now re-enable visibles
        if top:
            ax.xaxis.tick_top()
        if bottom:
            ax.xaxis.tick_bottom()
        if left:
            ax.yaxis.tick_left()
        if right:
            ax.yaxis.tick_right()

    emotion = [word[0] for word in sorted_emotion_corpus2_to_corpus1_ratios[:20]]
    ratio = [round(freq[1], 2) for freq in sorted_emotion_corpus2_to_corpus1_ratios[:20]]
    plt.figure(figsize=(5, 8))
    pos = np.arange(len(emotion))
    plt.title(corpus2_name + ' / ' + corpus1_name, horizontalalignment='center')
    barlist = plt.barh(pos, ratio)  # the plt.plot function returns certain info, e.g.,
    # the values for bars or lines or whatever is plotted. and these values are now put
    # into barlist object so i can use them below, e.g., loop through them and give the
    # bars some new attributes

    #add the numbers to the side of each bar
    for p, e, r in zip(pos, emotion, ratio):
        plt.annotate(str(r), xy=(r + .25, p + .5), va='center')

    #shade the bars based on intensity. it's looping through these values for the bars
    # and redrawsor re-does the bar with the new attribute
    for i in range(len(emotion)):
        the_emotion = emotion[i]
        valence = corresponding_root_to_ratings_dict[the_emotion][0]
        arousal = corresponding_root_to_ratings_dict[the_emotion][1]
        intensity_1 = corresponding_root_to_ratings_dict[the_emotion][2]
        intensity_2 = (np.abs(valence)/4) * (arousal*10)
        intensity_3 = arousal * np.sqrt(np.abs(valence/4))   #hmmm, this isn't bad. good for now.
        if corresponding_root_to_ratings_dict[the_emotion][0] > 0:
            barlist[i].set_color((0.0, 0.85, 0.0))
            barlist[i].set_alpha(intensity_3 / .7)
            print intensity_3
        else:
            barlist[i].set_color((0.99, 0.0, 0.0))
            barlist[i].set_alpha(intensity_3 / .7)
            print intensity_3
    #above, i like using arousal. might not be as good for
    #pos words in the opposite graph, though?

    #cutomize ticks
    ticks = plt.yticks(pos + .5, emotion, fontsize=18)
    xt = plt.xticks()[0]
    plt.xticks(xt, [' '] * len(xt))
    #minimize chartjunk
    remove_border(left=False, bottom=False)
    plt.grid(axis = 'x', color ='white', linestyle='-')
    #set plot limits
    plt.ylim(pos.max() + 1, pos.min() - 1)
    plt.xlim(0, 4)
    plt.tight_layout()  #this keeps words from getting cut off
    plt.savefig('static_test/corpus2_to_corpus1.png')  #will need to change this so saves to static folder



#plot_ratios_corpus2_to_corpus1('Dreams', 'Real Life', sorted_emotion_corpus2_to_corpus1_ratios, corresponding_root_to_ratings_dict)


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
    plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, corpus1_to_corpus2_ratios, corresponding_root_to_ratings_dict)
    plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, corpus2_to_corpus1_ratios, corresponding_root_to_ratings_dict)



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

#corpuses_to_plot(dream_corpus_clean_2, waking_corpus_clean_2, 'Dreams', 'Real-life', clore_and_storm_Mar19_dict)


#to get pickled dicts:
with open('root_to_variations_dict.pkl', 'r') as picklefile:
    root_to_variations_dict = pickle.load(picklefile)

with open('root_to_ratings_dict.pkl', 'r') as picklefile:
    root_to_ratings_dict = pickle.load(picklefile)

with open('corresponding_root_to_ratings_dict.pkl', 'r') as picklefile:
    corresponding_root_to_ratings_dict = pickle.load(picklefile)

len(corresponding_root_to_ratings_dict)
corresponding_root_to_ratings_dict['crying']
root_to_variations_dict['crying']


corpuses_to_plot(dream_corpus_clean_2, waking_corpus_clean_2, 'Dreams', 'Real-life', root_to_variations_dict)


##############################################################################
# this section: work on getting words assoc with emos
# and sentences with emos in them
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from sklearn.feature_extraction import text

# plan:
# within each corpus, divide up sentences for ea emo word
# within ea corpus, make dict that has each emo word as key and all sentences
#     from that corpus w the word as values. 
# treat each set of values/sentences as a doc. and tf-idf vectorize
#     to see how tf-idf scores for each word differ by emotion and corpus
# then can list the top tf-idf words for ea emotion within ea corpus
# if do it this way, doing it twice, once for ea corpus

# play around w only looking at tf-idfs for nouns. and verbs.
# could try presenting the highest and lowest ratios between corpora

# step 1:
# take corpus1. merge into one string doc. then sentence tokenize.
# then take root_to_variations_dict and loop through each set of variations 
# to see if any of them are in ea sent.
# if so, put sentence in new dict as a value for that root emo key.


len(dream_corpus_clean_2)

# create list of all emo variations (to use in stopwords):
def create_all_emo_variations_list(root_to_variations_dict):
    all_emo_variations = []
    for variation_set in root_to_variations_dict.values():
        all_emo_variations += variation_set
    return all_emo_variations

all_emo_variations = create_all_emo_variations_list(root_to_variations_dict)
len(set(all_emo_variations))


# loop through emo_to_variations_dict and loop through dreams in corpus so
# doing ea dream individually. because if change it later so that taking 
# two adjacent sentences too, then would need to do this part by dream
# Q: do i want to see what words are uniquely associated with a given 
# emotion across the two corpora? or see what words are uniquely assoc
# with each emotion uniquely for that corpora? I think that second one is 
# more interesting for the dream database. 

# this code is super slow. use code below. does same thing.
#tokenizer = TreebankWordTokenizer()
#root_to_sentences_dream_dict = defaultdict(list)
#for root in root_to_variations_dict:
#    for dream in dream_corpus_clean_2[:5]:
#        sentences_in_dream = sent_tokenize(dream)
#        for sentence in sentences_in_dream:
#            words_in_sent = tokenizer.tokenize(sentence)
#            if bool(set(root_to_variations_dict[root]) & set(words_in_sent)):
#                root_to_sentences_dream_dict[root].append(sentence)

# takes one corpus and returns a dict of root emos to sentences containing that emo:
def corpus_to_root_to_sentences(corpus_clean, root_to_variations_dict):
    tokenizer = TreebankWordTokenizer()
    root_to_sentences_dict = defaultdict(list)
    for dream in corpus_clean[:2]:
        sentences_in_dream = sent_tokenize(dream)
        for sentence in sentences_in_dream:
            words_in_sent = tokenizer.tokenize(sentence)
            for root in root_to_variations_dict:
                if bool(set(root_to_variations_dict[root]) & set(words_in_sent)):
                    root_to_sentences_dict[root].append(sentence)
    return root_to_sentences_dict

root_to_sentences_dream_dict = corpus_to_root_to_sentences(dream_corpus_clean_2, root_to_variations_dict)
len(root_to_sentences_dream_dict)
                

# create list of emotion docs, i.e., where each string is comprised of all the 
# sentences that contain a particular emotion word. neeed keep track of which 
# emo goes with which list item too. put keys/roots in a list too, in same order
def create_sentences_w_emo_list_and_emo_list(root_to_sentences_dict):
    combined_sent_around_emo_docs = []
    corresponding_emo_list = []
    for root in root_to_sentences_dict:
        combined_sent_around_emo_docs.append(' '.join([sentence for sentence in root_to_sentences_dict[root]]))
        corresponding_emo_list.append(root)
    return combined_sent_around_emo_docs, corresponding_emo_list

combined_sent_around_emo_docs, emo_list = create_sentences_w_emo_list_and_emo_list(root_to_sentences_dream_dict)
len(emo_list)
len(combined_sent_around_emo_docs)
root_to_sentences_dream_dict['crazy']
root_to_sentences_dream_dict.keys()[0]


# addition to can tfidf-vectorize both corp as same time.
# returns list of docs (w each doc comprised of sentences w an emo word in a corpus)
# and the emo list corresponding to those docs. but these lists have both info from
# corpus 1 and corpus 2. 
def create_sentences_w_emo_and_create_emo_list_two_corp(corpus1, corpus2):
    root_to_sentences_dict_corp1 = corpus_to_root_to_sentences(corpus1, root_to_variations_dict)    
    root_to_sentences_dict_corp2 = corpus_to_root_to_sentences(corpus2, root_to_variations_dict)    
    combined_sent_around_emo_docs1, emo_list1 = create_sentences_w_emo_list_and_emo_list(root_to_sentences_dict_corp1)
    combined_sent_around_emo_docs2, emo_list2 = create_sentences_w_emo_list_and_emo_list(root_to_sentences_dict_corp2)
    combined_sent_around_emo_docs12 = combined_sent_around_emo_docs1 + combined_sent_around_emo_docs2
    emo_list1_change = [emo + '1' for emo in emo_list1]
    emo_list2_change = [emo + '2' for emo in emo_list2]
    emo_list12 = emo_list1_change + emo_list2_change
    return combined_sent_around_emo_docs12, emo_list12


# build f here to keep only nouns, or verbs, in combined_sent_around_emo_docs
# and can call it or not in the master f. this should take combined_sent_around_emo_docs
# and return similar thing but just nouns or verbs inside
def only_NN_combined_sent_with_emo_docs(combined_sent_around_emo_docs):
    only_NN_combined_sent_w_emo_docs =[]
    for doc in combined_sent_around_emo_docs:
        textblob_doc = TextBlob(doc)
        only_part_of_speech = ' '.join([word_tag[0] for word_tag in textblob_doc.tags if word_tag[1] == 'NN' or word_tag[1] == 'NNS' or word_tag[1] == 'NNP' or word_tag[1] == 'NNPS']) 
        only_NN_combined_sent_w_emo_docs.append(only_part_of_speech)  
    return only_NN_combined_sent_w_emo_docs

only_NN_combined_sent_w_emo_docs = only_NN_combined_sent_with_emo_docs(combined_sent_around_emo_docs)
len(only_NN_combined_sent_w_emo_docs)


def only_VB_combined_sent_with_emo_docs(combined_sent_around_emo_docs):
    only_VB_combined_sent_w_emo_docs =[]
    for doc in combined_sent_around_emo_docs:
        textblob_doc = TextBlob(doc)
        only_part_of_speech = ' '.join([word_tag[0] for word_tag in textblob_doc.tags if word_tag[1] == 'VB' or word_tag[1] == 'VBD' or word_tag[1] == 'VBG' or word_tag[1] == 'VBN' or word_tag[1] == 'VBP' or word_tag[1] == 'VBZ']) 
        only_VB_combined_sent_w_emo_docs.append(only_part_of_speech)  
    return only_VB_combined_sent_w_emo_docs

only_VB_combined_sent_w_emo_docs = only_VB_combined_sent_with_emo_docs(combined_sent_around_emo_docs)



def tfidf_vectorize(all_emo_variations, combined_sent_around_emo_docs):
    # add all emotion words to stoplist    
    my_words = set(all_emo_variations)
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    #vectorizer = TfidfVectorizer(stop_words="english")  #orig text just using english stopwords
    vectorizer = TfidfVectorizer(stop_words=set(my_stop_words))  # add all emo words in dict as stop words
    words_around_emo_vectors = vectorizer.fit_transform(combined_sent_around_emo_docs)  #this is a list of strings.
    #vectorizer.get_feature_names()[5]  # gives names of words 
    return words_around_emo_vectors, vectorizer

words_around_emo_vectors, vectorizer = tfidf_vectorize(all_emo_variations, combined_sent_around_emo_docs)


# get terms in firs doc sorted by tf-idf
#terms = np.array(vectorizer.get_feature_names())
#terms_for_first_doc = zip(terms, words_around_emo_vectors.toarray()[0])  #the 0 is giving back terms for first doc
#sorted_terms_for_first_doc = sorted(terms_for_first_doc, key=lambda tup: tup[1], reverse=True)  #this sorts by the 2nd item in the tuple, the tf-idf score
#sorted_terms_for_first_doc[:10]
# ** this seemed to work. compare these terms_for_first_doc with another method
# yeah, checked against old method. this seems to work. use it


# get (terms, tf-idf) tuples sorted by tf-idf and put into root emo to (term, tf-idf) dict
def create_emo_to_tfidf__term_dict(vectorizer, combined_sent_around_emo_docs, words_around_emo_vectors, emo_list):
    root_to_tfidf_terms_dict = defaultdict(list)
    terms = np.array(vectorizer.get_feature_names())
    for i in range(len(combined_sent_around_emo_docs)):
        # the i is giving back info for first doc. so think getting the toarray is giving tf-idfs assoc with each term       
        terms_tfidf_for_doc = zip(terms, words_around_emo_vectors.toarray()[i])  
        sorted_terms_for_doc = sorted(terms_tfidf_for_doc, key=lambda tup: tup[1], reverse=True)  #this sorts by the 2nd item in the tuple, the tf-idf score
        root = emo_list[i]
        root_to_tfidf_terms_dict[root].append(sorted_terms_for_doc)
    return root_to_tfidf_terms_dict


root_to_tfidf_terms_dict = create_emo_to_tfidf__term_dict(vectorizer, combined_sent_around_emo_docs, words_around_emo_vectors, emo_list)
len(root_to_tfidf_terms_dict)



# create master f to run all this code above. 
#takes the corpus (e.g., of dreams) and returns a dict of 
# emo keys to terms assoc w those emos and tf-idfs of those terms
def master_corpus_to_emo_to_tfidf_term_dict(corpus, root_to_variations_dict):
    all_emo_variations = create_all_emo_variations_list(root_to_variations_dict)   
    root_to_sentences_dict = corpus_to_root_to_sentences(corpus, root_to_variations_dict)
    combined_sent_around_emo_docs, emo_list = create_sentences_w_emo_list_and_emo_list(root_to_sentences_dict)

    #if want to do just nouns or verbs, de-comment one of these below
    #combined_sent_around_emo_docs = only_NN_combined_sent_with_emo_docs(combined_sent_around_emo_docs)
    #combined_sent_around_emo_docs = only_VB_combined_sent_with_emo_docs(combined_sent_around_emo_docs)

    words_around_emo_vectors, vectorizer = tfidf_vectorize(all_emo_variations, combined_sent_around_emo_docs)
    root_to_tfidf_terms_dict = create_emo_to_tfidf__term_dict(vectorizer, combined_sent_around_emo_docs, words_around_emo_vectors, emo_list)
    return root_to_tfidf_terms_dict

root_to_tfidf_terms_dict = master_corpus_to_emo_to_tfidf_term_dict(dream_corpus_clean_2, root_to_variations_dict)


# gives tfidfs for each emo labeled as either belonging to corpus1 or corpus2
def alt_master_corpus_to_emo_to_tfidf_term_dict(corpus1, corpus2, root_to_variations_dict):
    all_emo_variations = create_all_emo_variations_list(root_to_variations_dict)       
    combined_sent_around_emo_docs12, emo_list12 = create_sentences_w_emo_and_create_emo_list_two_corp(corpus1, corpus2)

    #if want to do just nouns or verbs, de-comment one of these below
    #combined_sent_around_emo_docs12 = only_NN_combined_sent_with_emo_docs(combined_sent_around_emo_docs12)
    #combined_sent_around_emo_docs12 = only_VB_combined_sent_with_emo_docs(combined_sent_around_emo_docs12)

    words_around_emo_vectors, vectorizer = tfidf_vectorize(all_emo_variations, combined_sent_around_emo_docs12)
    root_to_tfidf_terms_dict = create_emo_to_tfidf__term_dict(vectorizer, combined_sent_around_emo_docs12, words_around_emo_vectors, emo_list12)
    return root_to_tfidf_terms_dict

root_to_tfidf_terms_dict_combo_corpora = alt_master_corpus_to_emo_to_tfidf_term_dict(dream_corpus_clean_2, waking_corpus_clean_2, root_to_variations_dict)


# now can explore this dict and try nn ony and vb only, and see what else I can do
# to make this interesting. ratio again? to compare ea corpus? 
# to compare two emos? -- could do that too.







# see how to tweak so can run it for both corpora at same time.
# what if i combo both corpora in to 'corpus'? don't think that works? because...






# next steps:
# do for all of corpus
# consider doing tf-idf for both corpuses at same time so can distinguish
# between coropuses. and more is probaby better.
# include emo variations as stop words
# use just nouns and/or just verbs
# somehow get rid of non-words as early as possible in process, 
# so don't get used for tf-idf
# need to be able to go back into root_to_sentences_dict and grab a couple
# sentences with a particular word. i guess, get top tf-idf word for a particular
# emotion and then go to the sentences under that key in dict and say, 
# give me all sentences wiht this word and print one that's a reasonable
# lenght? or just print the first one. or first two.
# ((and then always can try just counting the words/nouns/verbs that appear 
# most frequently around the emo word. and perhaps doing ratio. but skip 
# this for now, now that i have the tf-idf going.))
# do i also need to think about whether to group certain nouns or verbs
# together? e.g., do something like did with emo words -- lump variations
# of same words together? could do this with irmak's wordnet approach to 
# sort of get synonyms. but this seems more complicated? wait on this.
# first play around with more basic stuff to see how it looks.











# see if this does the same as above -- finding top tf-idf words for a doc:
#c_matrix = coo_matrix(emotion_vectors)
#
#def print_representative_words(number_docs, number_words, c_matrix, vectorizer):
#    for i in range(number_docs):
#        #iterate through sparse/coo matrix to get tuples:
#        variable_list = [(index, tfidf) for row, index, tfidf in zip(c_matrix.row, c_matrix.col, c_matrix.data) if row == i]
#        #sort tuples based on tf-idf score:
#        variables_sorted_by_tfidf = sorted(variable_list, key=lambda tup: tup[1], reverse=True)
#        word_list = []
#        for x in range(number_words):
#            max_index_in_cluster = variables_sorted_by_tfidf[x][0]
#            word = vectorizer.get_feature_names()[max_index_in_cluster]     
#            word_list.append(word)
#        print 'Words representative of document {}: '.format(i+1), word_list
#        print
#
#print_representative_words(1, 5, c_matrix, vectorizer)



# draw from unsupervised learning 1 ipy nb to get the most important
# tf-idf words for each emo doc
c_matrix = coo_matrix(emotion_vectors)
row_1 = c_matrix.getrow(0)
sorted_row_1 = np.argsort(row_1)
sorted_row_1 = np.sort(row_1)
sorted_row_1 = sorted(row_1)

#gives max of a row
row_maximum = max(c_matrix.getrow(18).data)

# the row col and data give the row#, col# and data point for each data point in sparse matrix
count = 0
for i, j, v in zip(c_matrix.row, c_matrix.col, c_matrix.data):  
    print i, j, v
    count += 1
    if count == 10:
        break

long_string = ' '.join([dream for dream in dream_corpus_clean_2[:3]]) 
# this is a prob if later want to take adjoining sentences because may be 
# taking them from adjoining dreams

text = [sentence for sentence in dream_corpus_clean_2[:3]]

text = "Hello. How are you, dear sir? Are you well, Mr. Sirer? Here: drink this! It will make you feel better."

sentences = sent_tokenize(text[0])  #needs to take one long string doc
sentences


















###################################################################################
# testing out to look at bayes factors, p-vals, etc.

corpus_wake_lower = corpus_lowercase(waking_corpus_clean_2)
len(corpus_wake_lower)
#corpus_lower_spelling = corpus_spelling_correct(corpus_lower)
#len(corpus_lower_spelling)  #this took forever to compute!!! elim from function for now.
corpus_wake_simplify_emo_words = replace_emo_words_w_root(corpus_wake_lower, clore_and_storm_Mar19_dict)
len(corpus_wake_simplify_emo_words)  #this is each text descriptions 
emotion_to_count_wake_dict = count_docs_w_ea_emotion(corpus_wake_simplify_emo_words, clore_and_storm_Mar19_dict)
len(emotion_to_count_wake_dict)
#alphabetical_emotions_w_counts_list = sort_emotion_counts_alphabetically(emotion_to_count_dict)
#len(alphabetical_emotions_w_counts_list)
#alphabetical_emotions_w_counts_list[:15]

corpus_dream_lower = corpus_lowercase(dream_corpus_clean_2)
len(corpus_dream_lower)
corpus_dream_simplify_emo_words = replace_emo_words_w_root(corpus_dream_lower, clore_and_storm_Mar19_dict)
len(corpus_dream_simplify_emo_words)  #this is each text descriptions 
emotion_to_count_dream_dict = count_docs_w_ea_emotion(corpus_dream_simplify_emo_words, clore_and_storm_Mar19_dict)
len(emotion_to_count_dream_dict)


# ok, this is what i want: for each emotion word (key), the values are the list
# of 0s and 1s representing whether each doc doesn't have or has the emo word
keys_list = emotion_to_count_wake_dict.keys()
truncated_keys_list = keys_list[:10]
for key in truncated_keys_list:    
    print len(emotion_to_count_wake_dict[key])
    print


# get the wake and dream dicts into df
import pandas as pd
waking_df = pd.DataFrame(emotion_to_count_wake_dict)
waking_df.head()
waking_df.describe()
len(waking_df)
waking_df['description_type'] = 'waking'

dream_df = pd.DataFrame(emotion_to_count_dream_dict)
dream_df.head()
dream_df.describe()
len(dream_df)
dream_df['description_type'] = 'dream'

# merge the two dfs
df1 = pd.DataFrame({'a': [1, 3, 4], 'b': [4, 1, 0]})
df2 = pd.DataFrame({'a': [11, 13, 14], 'b': [14, 11, 10]})
df1_df2 = pd.concat([df1, df2], ignore_index=True)

dream_and_waking_df = pd.concat([waking_df, dream_df], ignore_index=True)
dream_and_waking_df.tail()
emos = dream_and_waking_df.columns
len(emos)
emos = list(emos)
emos.pop(-1)

import thinkstats2
import statsmodels.formula.api as sm #need this 'formula' api to use the R-style code. seems simpler.

results1 = sm.ols(formula = 'scary ~ description_type', data=dream_and_waking_df).fit()
print results1.summary()
p_value = results1.pvalues[1]

dir(results1)

def bayesFact(resultsObj):
    n = resultsObj.nobs
    df_effect = resultsObj.df_model
    df_error = resultsObj.df_resid
    MSE_effect = resultsObj.mse_model
    MSE_error = resultsObj.mse_resid
    SS_effect = MSE_effect * df_effect
    SS_error = MSE_error * df_error
    SS_total = SS_effect + SS_error
    deltaBIC = (n * np.log(SS_error / SS_total)) + (df_effect * np.log(n))
    BF01 = np.exp(deltaBIC / 2)
    pOfH0givenD = BF01 / (1 + BF01)
    pOfH1givenD = 1 - pOfH0givenD
    pTrue = pOfH1givenD / (pOfH1givenD + pOfH0givenD) 
    return pOfH1givenD / pOfH0givenD 

bayes_factor_excel = bayesFact(results1)
dir(results1)

# my bayes factor computation function

dream_and_waking_df_vars = dream_and_waking_df[['description_type', 'scary']]
waking_df_var = dream_and_waking_df_vars[dream_and_waking_df_vars['description_type'] == 'waking']
dream_df_var = dream_and_waking_df_vars[dream_and_waking_df_vars['description_type'] == 'dream']
waking_list = waking_df_var['scary'].values
dream_list = dream_df_var['scary'].values

np.mean(waking_list)
np.mean(dream_list)


#def bootstrap_diffs_list_H1(list1, list2):
#    bootstrapped_differences_list_H1 = []
#    for i in range(1000):
#        sample1 = []
#        for i in range(len(list1)):
#            sample1.append(random.choice(list1))
#        sample2 = []
#        for i in range(len(list2)):  #148
#            sample2.append(random.choice(list2))
#        mean_difference = np.mean(sample2) - np.mean(sample1)
#        bootstrapped_differences_list_H1.append(mean_difference)
#    return bootstrapped_differences_list_H1
#
#def bootstrap_diffs_list_H0(list1, list2):
#    list_pooled = list1 + list2
#    bootstrapped_differences_list_H0 = []
#    for i in range(1000):
#        sample1 = []
#        for i in range(len(list1)):
#            sample1.append(random.choice(list_pooled))
#        sample2 = []
#        for i in range(len(list2)):  #148
#            sample2.append(random.choice(list_pooled))
#        mean_difference = np.mean(sample2) - np.mean(sample1)
#        bootstrapped_differences_list_H0.append(mean_difference)
#    return bootstrapped_differences_list_H0
#
#def prob_data_given_H(bootstrapped_differences_list):
#    H_pdf = thinkstats2.EstimatedPdf(bootstrapped_differences_list)
#    prob_of_observed_diff_given_H = H_pdf.Density(observed_diff)  
#    return prob_of_observed_diff_given_H
#
#def compute_bayes_factor(prob_of_observed_diff_given_H1, prob_of_observed_diff_given_H0):
#    bayes_factor = prob_of_observed_diff_given_H1 / prob_of_observed_diff_given_H0
#    bayes_factor_pct = bayes_factor / (1 + bayes_factor)
#    return (bayes_factor, bayes_factor_pct)
#
#def main_bayes(list1, list2):
#    boot_diffs_list_H1 = bootstrap_diffs_list_H1(list1, list2)
#    boot_diffs_list_H0 = bootstrap_diffs_list_H0(list1, list2)
#    prob_data_given_H1 = prob_data_given_H(boot_diffs_list_H1)
#    prob_data_given_H0 = prob_data_given_H(boot_diffs_list_H0)
#    bayes_info = compute_bayes_factor(prob_data_given_H1, prob_data_given_H0)
#    print 'bayes factor: {}, bayes factor pct: {}'.format(round(bayes_info[0][0], 2), round(bayes_info[1][0], 2))    
#    return bayes_info
#
#bayes_info_test = main_bayes(list(waking_list), list(dream_list))
##this doesn't work at all!!!

# to bootstrap p-value:
def get_bootstrapped_diffs_list(list_pooled):
    bootstrapped_differences_list = []
    for i in range(500):
        sample1 = []
        for i in range(122):
            sample1.append(random.choice(list_pooled))
        sample2 = []
        for i in range(148):  #148
            sample2.append(random.choice(list_pooled))
        mean_difference = np.mean(sample2) - np.mean(sample1)
        bootstrapped_differences_list.append(mean_difference)
    return bootstrapped_differences_list

def bootstrapped_diffs_to_pvalue(bootstrapped_differences_list, observed_diff):
    observed_diff_bigger = 0
    for diff in bootstrapped_differences_list:
        if np.abs(diff) > np.abs(observed_diff):  #this abs val is key to getting the two-tailed, i think
            observed_diff_bigger += 1
    p_frequentist = observed_diff_bigger / 500
    return p_frequentist

# this produces the exact p-value as above. but it's diff than the p-val
# produced by statsmodels. need to try this bootstrapping approach on a 
# normal dataset w continuous dv that's normally distributed and see if 
# it produces basically the same values there.
def bootstrapped_diffs_to_pvalue_2(bootstrapped_differences_list, observed_diff):
    observed_diff_bigger = 0
    observed_diff_smaller = 0
    for diff in bootstrapped_differences_list:
        if diff > np.abs(observed_diff):  
            observed_diff_bigger += 1
        if diff < -1*(np.abs(observed_diff)):
            observed_diff_smaller += 1
    extreme_vals = observed_diff_bigger + observed_diff_smaller
    p_frequentist = extreme_vals / 500
    return p_frequentist
# http://people.duke.edu/~ccc14/pcfb/analysis.html    

# problem - this is returning something that corr with regular p-val
# but it's a lot more conservative. so not the same thing.
# wonder how it's diff. and wonder if this would be a better test for me to use
# unless it's because these data are wacky. try on more normal data
def get_pvalue_bootstrapped(list_pooled, observed_diff):
    bootstrapped_diffs = get_bootstrapped_diffs_list(list_pooled)
    p_value1 = bootstrapped_diffs_to_pvalue(bootstrapped_diffs, observed_diff)
    p_value2 = bootstrapped_diffs_to_pvalue_2(bootstrapped_diffs, observed_diff)
    return [p_value1, p_value2]


# to do stats anys giving p-val and bayes factor

waking_df = dream_and_waking_df[dream_and_waking_df['description_type'] == 'waking']
dream_df = dream_and_waking_df[dream_and_waking_df['description_type'] == 'dream']


dict_pval_bf = {'emotion': [], 'pvalue': [], 'bf': [], 'ratio5': [], 'ratio10': [], 'ratio15': [], 'difference': [], 'pvalue_boot1': [], 'pvalue_boot2':[]}

for emo in emos:
    # to get p-val:    
    results1 = sm.ols(formula = '{} ~ description_type'.format(emo), data=dream_and_waking_df).fit()
    p_value = results1.pvalues[1]
    # to get bf:
    bayes_factor_excel = bayesFact(results1)
    # to get bootstrapped p-vals:
    waking_list = list(waking_df[emo].values)
    dream_list = list(dream_df[emo].values)
    pooled_list = waking_list + dream_list
    observed_diff = np.mean(dream_list) - np.mean(waking_list)
    p_values_bootstrapped = get_pvalue_bootstrapped(pooled_list, observed_diff)
    p_value_bootstrapped_1 = p_values_bootstrapped[0]
    p_value_bootstrapped_2 = p_values_bootstrapped[1]
    
    # to get ratio:
    ratio_dreams_to_waking5 = (np.mean(dream_list)*100 + 5) / (np.mean(waking_list)*100 + 5)
    ratio_dreams_to_waking10 = (np.mean(dream_list)*100 + 10) / (np.mean(waking_list)*100 + 10)
    ratio_dreams_to_waking15 = (np.mean(dream_list)*100 + 15) / (np.mean(waking_list)*100 + 15)
    # get pct diff:        
    difference_dream_waking = (np.mean(dream_list)*100) - (np.mean(waking_list)*100)
    # put into dict:
    dict_pval_bf['emotion'].append(emo)
    dict_pval_bf['pvalue'].append(p_value)
    dict_pval_bf['pvalue_boot1'].append(p_value_bootstrapped_1)
    dict_pval_bf['pvalue_boot2'].append(p_value_bootstrapped_2)
    dict_pval_bf['bf'].append(bayes_factor_excel)
    dict_pval_bf['ratio5'].append(ratio_dreams_to_waking5)
    dict_pval_bf['ratio10'].append(ratio_dreams_to_waking10)
    dict_pval_bf['ratio15'].append(ratio_dreams_to_waking15)
    dict_pval_bf['difference'].append(difference_dream_waking)



len(dict_pval_bf['emotion'])
len(emos)
len(waking_list)
len(waking_df.columns)
len(pooled_list)

dir(results1)
help(results1)
dir(sm)

#SWEET WORKING. FOR FOR ALL EMOS. THEN TURN DICT INTO DF.
#THEN LOOK AT SCATTERPLOT. MAY HAVE TO DO DROP NA?

df_effects = pd.DataFrame(dict_pval_bf)    
df_effects.head()

df_effects.plot(kind='scatter', x='pvalue', y='pvalue_boot1')


df_effects.plot(kind='scatter', x='pvalue', y='difference')
plt.xlim(-.02, .8)
plt.ylim(-.1, .8)


plt.plot(df_effects['pvalue'], df_effects['bf'], 'ro')
plt.ylim(0, 1)
plt.xlim(0, .3)

#plt.plot(df_effects['pvalue_boot'], df_effects['bf'], 'ro')
#plt.ylim(0, 1)

plt.plot(df_effects['pvalue'], df_effects['ratio'], 'ro')
plt.xlim(-.01, .06)

df_effects.plot(kind='scatter', x='pvalue', y='ratio')

#df_effects.plot(kind='scatter', x='pvalue_boot', y='ratio')

df_effects.plot(kind='scatter', x='bf', y='ratio')
plt.xlim(0, .3)


df_effects.plot(kind='scatter', x='difference', y='ratio5')
plt.xlim(-10, 10)

df_effects.plot(kind='scatter', x='difference', y='ratio10')
df_effects.plot(kind='scatter', x='difference', y='ratio15')


df_effects.plot(kind='scatter', x='pvalue', y='ratio15')



#############################################################################
# get good list/dictonary of emo words








 
