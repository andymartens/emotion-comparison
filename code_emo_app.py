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


#plot  -  TURN INTO FUNCTION:
def plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios):
    X = [word[0] for word in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
    Y = [freq[1] for freq in sorted_emotion_corpus1_to_corpus2_ratios[:25]]
    fig = plt.figure(figsize=(10, 6))  #add this to set resolution: , dpi=100
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

# alt version of above function, to play with:
# this is working ok. need to reverse order so biggest vaules on top.
# def plot_ratios_corpus1_to_corpus2(corpus1_name, corpus2_name, sorted_emotion_corpus1_to_corpus2_ratios):
X = [word[0] for word in sorted_emotion_corpus1_to_corpus2_ratios[:20]]
Y = [freq[1] for freq in sorted_emotion_corpus1_to_corpus2_ratios[:20]]
plt.figure(figsize=(8, 12))  #add this to set resolution: , dpi=100
barlist = plt.barh(np.array(range(len(X))), np.array(Y))
sns.despine(left=True)
plt.title('Emotion-words Most Representative of ' + corpus1_name, fontsize=17)
#plt.xticks(rotation=75)
plt.yticks(np.array(range(len(X))) + .5, np.array(X), fontsize=15)
plt.xlim(1, 3.05)
plt.xlabel('Frequency in {} relative to {}'.format(corpus1_name, corpus2_name), fontsize=15)
#this tight_layout method fixes problem of x-axis labels cut off in saved figure:    
for i in range(len(X)):
    if Y[i] < 1.6:    
        barlist[i].set_color((0.0, 0.6, 0.0))
        barlist[i].set_alpha(Y[i]/3)
    else:    
        barlist[i].set_color((0.6, 0.0, 0.0))
        barlist[i].set_alpha(Y[i]/3)    
plt.tight_layout()
plt.savefig('static_test/corpus1_to_corpus2.png')


# get Xs and Ys to play w:
corpus1_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(dream_corpus_clean_2, root_to_variations_dict)
corpus2_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(waking_corpus_clean_2, root_to_variations_dict)
corpus1_to_corpus2_ratios = get_emotion_corpus1_to_corpus2_ratios(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list)
corpus2_to_corpus1_ratios = get_emotion_corpus2_to_corpus1_ratios(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list)
sorted_emotion_corpus1_to_corpus2_ratios = corpus1_to_corpus2_ratios
sorted_emotion_corpus2_to_corpus1_ratios = corpus2_to_corpus1_ratios
corpus1_name = 'Dreams'
corpus2_name = 'Real-life'


# above works ok. but compare to this syntax from web:
from matplotlib import rcParams

#colorbrewer2 Dark2 qualitative color table
#dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
#rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'  #this is the background color of the grid area
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
#rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'


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
intensity = [.2, .7, .7, .9, .1, .8, .7, .6, .1, .5, 
             .4, .1, .6, .9, .4, .7, .9, .8, .7, .6]

grad = pd.DataFrame({'ratio' : ratio, 'emotion': emotion, 'intensity': intensity})
plt.figure(figsize=(3, 8))
#change = grad.change[grad.change > 0]  #in future step, just create one long list of ratios and select the top 20 here (and bottom 20 when do other/second graph)
#city = grad.city[grad.change > 0]
#intensity = grad.intensity[grad.change > 0]
#ratio = grad.ratio
#emotion = grad.emotion
#intensity = grad.intensity

pos = np.arange(len(emotion))
plt.title('1995-2005 Change in HS graduation rate')
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
    if corresponding_root_to_ratings_dict[the_emotion][0] > 0:    
        barlist[i].set_color((0.0, 0.9, 0.0))
        barlist[i].set_alpha(corresponding_root_to_ratings_dict[the_emotion][2] / 3)
    else:    
        barlist[i].set_color((0.99, 0.0, 0.0))
        barlist[i].set_alpha(np.abs(corresponding_root_to_ratings_dict[the_emotion][2]) / 3)
plt.savefig('static_test/corpus1_to_corpus2_ALT.png')

#cutomize ticks
ticks = plt.yticks(pos + .5, emotion)
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))
#minimize chartjunk
remove_border(left=False, bottom=False)
plt.grid(axis = 'x', color ='white', linestyle='-')
#set plot limits
plt.ylim(pos.max() + 1, pos.min() - 1)
plt.xlim(0, 3.5)
############################################################


intensity_list = [ratings[2] for ratings in corresponding_root_to_ratings_dict.values()]
np.nanmean(intensity_list)
np.nanmin(intensity_list)
np.nanmax(intensity_list)

arousal_list = [ratings[1] for ratings in corresponding_root_to_ratings_dict.values()]
np.nanmean(arousal_list)
np.nanmin(arousal_list)
np.nanmax(arousal_list)







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








 
