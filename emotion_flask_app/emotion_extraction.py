# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:49:44 2015

@author: charlesmartens
"""

from pattern import en  #don't think i need this anymore?
from pattern.en import conjugate, lemma, lexeme
from pattern.en import wordnet as wn  #don't think i need this?
import enchant  #for checking if actual word. may need this for tf-idf stuff
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns  #don't think i need this anymore?
import pickle
from matplotlib import rcParams



# Retrieve corresponding_root_to_ratings_dict:
with open('corresponding_root_to_ratings_dict.pkl', 'r') as picklefile:
    corresponding_root_to_ratings_dict = pickle.load(picklefile)


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
def replace_emo_words_w_root(corpus, root_to_variations_dict):
    corpus_replaced_emotions = []
    for report in corpus:
        for key in root_to_variations_dict.keys():
            for word in root_to_variations_dict[key]:
                report = report.replace(word, key)
        corpus_replaced_emotions.append(report)
    return corpus_replaced_emotions


#create dict where emotion_complete category is the key and the values are whether absent or present in each report
def count_docs_w_ea_emotion(corpus, root_to_variations_dict):
    count_of_ea_emotion_dict = defaultdict(list)
    for report in corpus:
        for emotion in root_to_variations_dict.keys():
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


#plot  -
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
    #intensity = [.2, .7, .7, .9, .1, .8, .7, .6, .1, .5,
    #             .4, .1, .6, .9, .4, .7, .9, .8, .7, .6]

    #grad = pd.DataFrame({'ratio' : ratio, 'emotion': emotion, 'intensity': intensity})
    plt.figure(figsize=(5, 8))
    #change = grad.change[grad.change > 0]  #in future step, just create one long list of ratios and select the top 20 here (and bottom 20 when do other/second graph)
    #city = grad.city[grad.change > 0]
    #intensity = grad.intensity[grad.change > 0]
    #ratio = grad.ratio
    #emotion = grad.emotion
    #intensity = grad.intensity

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
    plt.savefig('static/corpus1_to_corpus2.png')  #will need to change this so saves to static folder


# def plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios):
#     X = [word[0] for word in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
#     Y = [freq[1] for freq in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
#     fig = plt.figure(figsize=(15, 5))  #add this to set resolution: , dpi=100
#     sns.barplot(x = np.array(range(len(X))), y = np.array(Y))
#     sns.despine(left=True)
#     plt.title('Emotion-words Most Representative of ' + corpus1_name, fontsize=17)
#     plt.xticks(rotation=75)
#     plt.xticks(np.array(range(len(X))), np.array(X), rotation=75, fontsize=15)
#     plt.ylim(1, 3.05)
#     plt.ylabel('Frequency in {} relative to {}'.format(corpus1_name, corpus2_name), fontsize=15)
#     plt.tight_layout()
#     plt.savefig("static/corpus1_to_corpus2.png")


#plot  -
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
    #intensity = [.2, .7, .7, .9, .1, .8, .7, .6, .1, .5,
    #             .4, .1, .6, .9, .4, .7, .9, .8, .7, .6]

    #grad = pd.DataFrame({'ratio' : ratio, 'emotion': emotion, 'intensity': intensity})
    plt.figure(figsize=(5, 8))
    #change = grad.change[grad.change > 0]  #in future step, just create one long list of ratios and select the top 20 here (and bottom 20 when do other/second graph)
    #city = grad.city[grad.change > 0]
    #intensity = grad.intensity[grad.change > 0]
    #ratio = grad.ratio
    #emotion = grad.emotion
    #intensity = grad.intensity

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
    plt.savefig('static/corpus2_to_corpus1.png')  #will need to change this so saves to static folder

# def plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, sorted_emotion_corpus2_to_corpus1_ratios):
#     X = [word[0] for word in sorted_emotion_corpus2_to_corpus1_ratios[:25]]
#     Y = [freq[1] for freq in sorted_emotion_corpus2_to_corpus1_ratios[:25]]
#     fig = plt.figure(figsize=(15, 5))  #add this to set resolution: , dpi=100
#     sns.barplot(x = np.array(range(len(X))), y = np.array(Y))
#     sns.despine(left=True)
#     plt.title('Emotion-words Most Representative of ' + corpus2_name, fontsize=17)
#     plt.xticks(rotation=75)
#     plt.xticks(np.array(range(len(X))), np.array(X), rotation=75, fontsize=15)
#     plt.ylim(1, 3.05)
#     plt.ylabel('Frequency in {} relative to {}'.format(corpus2_name, corpus1_name), fontsize=15)
#     plt.tight_layout()
#     plt.savefig("static/corpus2_to_corpus1.png")


def corpus_to_alphabetical_emotion_counts(corpus, root_to_variations_dict):
    corpus_lower = corpus_lowercase(corpus)
    #corpus_lower_spelling = corpus_spelling_correct(corpus_lower)
    corpus_simplify_emo_words = replace_emo_words_w_root(corpus_lower, root_to_variations_dict)
    emotion_to_count_dict = count_docs_w_ea_emotion(corpus_simplify_emo_words, root_to_variations_dict)
    alphabetical_emotions_w_counts_list = sort_emotion_counts_alphabetically(emotion_to_count_dict)
    return alphabetical_emotions_w_counts_list


def plot_alphabetical_lists(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2, corpus1_name, corpus2_name, corresponding_root_to_ratings_dict):
    corpus1_to_corpus2_ratios = get_emotion_corpus1_to_corpus2_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2)
    corpus2_to_corpus1_ratios = get_emotion_corpus2_to_corpus1_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2)
    plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, corpus1_to_corpus2_ratios, corresponding_root_to_ratings_dict)
    plot_ratios_corpus2_to_corpus1(corpus1_name, corpus2_name, corpus2_to_corpus1_ratios, corresponding_root_to_ratings_dict)


##############################################################################
#master function -- takes input of corpuses and outputs 2 plots:
def corpuses_to_plot(corpus1, corpus2, corpus1_name, corpus2_name, root_to_variations_dict, corresponding_root_to_ratings_dict):
    corpus1_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(corpus1, root_to_variations_dict)
    corpus2_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(corpus2, root_to_variations_dict)
    plot_alphabetical_lists(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list, corpus1_name, corpus2_name, corresponding_root_to_ratings_dict)
###############################################################################


#corpuses_to_plot(dream_corpus_clean_2, waking_corpus_clean_2, 'Dreams', 'Real-life', clore_and_storm_Mar19_dict)
