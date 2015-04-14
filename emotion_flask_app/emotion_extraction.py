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


# in each set of text docs change all words in reports to lowercase
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
    plt.figure(figsize=(5, 6.5))
    pos = np.arange(len(emotion))
    #plt.title(corpus1_name + ' / ' + corpus2_name, horizontalalignment='center')
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
    plt.xlim(0, 5)

    plt.tight_layout(pad=0.01)  #this keeps words from getting cut off
    plt.savefig('static/corpus1_to_corpus2.png')  #will need to change this so saves to static folder


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
    plt.figure(figsize=(5, 6.5))
    pos = np.arange(len(emotion))
    #plt.title(corpus2_name + ' / ' + corpus1_name, horizontalalignment='center')
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
    plt.xlim(0, 5)

    plt.tight_layout(pad=.01)  #this keeps words from getting cut off
    plt.savefig('static/corpus2_to_corpus1.png')  #will need to change this so saves to static folder


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


################################################################################
#master function -- takes input of corpuses and outputs 2 plots:
def corpuses_to_plot(corpus1, corpus2, corpus1_name, corpus2_name, root_to_variations_dict, corresponding_root_to_ratings_dict):
    corpus1_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(corpus1, root_to_variations_dict)
    corpus2_alphabetical_counts_list = corpus_to_alphabetical_emotion_counts(corpus2, root_to_variations_dict)
    plot_alphabetical_lists(corpus1_alphabetical_counts_list, corpus2_alphabetical_counts_list, corpus1_name, corpus2_name, corresponding_root_to_ratings_dict)
###############################################################################
#corpuses_to_plot(dream_corpus_clean_2, waking_corpus_clean_2, 'Dreams', 'Real-life', clore_and_storm_Mar19_dict)





################################################################################
# functions for getting words associated with particular emotions and examples
# of text containing and specific emotions

from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from sklearn.feature_extraction import text
import enchant
import pickle
import numpy as np
from textblob import TextBlob
from collections import defaultdict


# Retrieve emotion-words to variations dict:
with open('root_to_variations_dict.pkl', 'r') as picklefile:
    root_to_variations_dict = pickle.load(picklefile)


def create_all_emo_variations_list(root_to_variations_dict):
    all_emo_variations = []
    for variation_set in root_to_variations_dict.values():
        all_emo_variations += variation_set
    return all_emo_variations

all_emo_variations = create_all_emo_variations_list(root_to_variations_dict)


def sentences_combo(sentences_in_doc, i):
    x = (i-1) >= 0
    y = (i+2) <= len(sentences_in_doc)
    if x and y:
        combo_sentences = sentences_in_doc[i-1] + ' ' + sentences_in_doc[i] + ' ' + sentences_in_doc[i+1]
        return combo_sentences
    elif x == False and y == True:
        combo_sentences = sentences_in_doc[i] + ' ' + sentences_in_doc[i+1]
        return combo_sentences
    elif x == True and y == False:
        combo_sentences = sentences_in_doc[i-1] + ' ' + sentences_in_doc[i]
        return combo_sentences
    elif x == False and y == False:
        combo_sentences = sentences_in_doc[i]
        return combo_sentences


# takes one corpus and returns a dict of root emos to sentences containing that emo
# set it up now so takes the sentence with emo and adjacent two sentences also
def corpus_to_root_to_sentences(corpus_clean, root_to_variations_dict):
    #tokenizer = TreebankWordTokenizer()
    root_to_sentences_dict = defaultdict(list)
    for doc in corpus_clean[:]:
        words_in_doc = TextBlob(doc).words
        for root in root_to_variations_dict:
            if bool(set(root_to_variations_dict[root]) & set(words_in_doc)):
                sentences_in_doc = sent_tokenize(doc)
                i = 0
                while i < len(sentences_in_doc):
                #for i in range(len(sentences_in_doc)):
                    if bool(set(root_to_variations_dict[root]) & set(TextBlob(sentences_in_doc[i]).words)):
                        combo_sentences = sentences_combo(sentences_in_doc, i)
                        root_to_sentences_dict[root].append(combo_sentences)
                        i = len(sentences_in_doc) + 1
                    else:
                        i = i + 1
    return root_to_sentences_dict


# create list of emotion docs, i.e., where each string is comprised of all the
# sentences that contain a particular emotion word. but made ea sentence a set of words, i.e., took out repeats
# to keep track of which emo goes with which list item, put keys/roots in a list too, in same order
def create_sentences_w_emo_list_and_emo_list(root_to_sentences_dict):
    combined_sent_around_emo_docs = []
    corresponding_emo_list = []
    for root in root_to_sentences_dict:
        # turn ea sentence into a set of words, so words dont count twice for one document, e.g., for one dream
        sentences = root_to_sentences_dict[root]
        sentences_unique_words = []
        for sentence in sentences:
            sentence_blob = TextBlob(sentence)
            bag_of_words_unique_in_sentence = list(set(sentence_blob.words))
            sentences_unique_words += bag_of_words_unique_in_sentence
        sentences_unique_words_joined = ' '.join(sentences_unique_words)
        combined_sent_around_emo_docs.append(sentences_unique_words_joined)
        corresponding_emo_list.append(root)
    return combined_sent_around_emo_docs, corresponding_emo_list


# addition to can tfidf-vectorize both corp as same time.
# returns list of docs (w each doc comprised of sentences w an emo word in a corpus)
# and the emo list corresponding to those docs. but these lists have both info from
# corpus 1 and corpus 2.
def create_sentences_w_emo_and_create_emo_list_two_corp(corpus1, corpus2):
    root_to_sentences_dict_corp1 = corpus_to_root_to_sentences(corpus1, root_to_variations_dict)
    root_to_sentences_dict_corp2 = corpus_to_root_to_sentences(corpus2, root_to_variations_dict)
    with open('root_to_sentences_dict_corp1.pkl', 'w') as picklefile:
        pickle.dump(root_to_sentences_dict_corp1, picklefile)
    with open('root_to_sentences_dict_corp2.pkl', 'w') as picklefile:
        pickle.dump(root_to_sentences_dict_corp2, picklefile)
    # comment out above two fs and uncomment below two fs to compare looking at sentences
    # with emos vs. ~300 characters that contain emos
    #root_to_sentences_dict_corp1 = corpus_to_root_to_sentences_alt(corpus1, root_to_variations_dict)
    #root_to_sentences_dict_corp2 = corpus_to_root_to_sentences_alt(corpus2, root_to_variations_dict)
    combined_sent_around_emo_docs1, emo_list1 = create_sentences_w_emo_list_and_emo_list(root_to_sentences_dict_corp1)
    combined_sent_around_emo_docs2, emo_list2 = create_sentences_w_emo_list_and_emo_list(root_to_sentences_dict_corp2)
    combined_sent_around_emo_docs12 = combined_sent_around_emo_docs1 + combined_sent_around_emo_docs2
    emo_list1_change = [emo + '1' for emo in emo_list1]
    emo_list2_change = [emo + '2' for emo in emo_list2]
    emo_list12 = emo_list1_change + emo_list2_change
    return combined_sent_around_emo_docs12, emo_list12


# create third option here that gives back both nouns AND verbs
def only_VB_NN_combined_sent_with_emo_docs(combined_sent_around_emo_docs):
    d = enchant.Dict("en_US")
    only_VB_NN_combined_sent_w_emo_docs =[]
    for doc in combined_sent_around_emo_docs:
        textblob_doc = TextBlob(doc)
        doc_lc_sing_real = ' '.join([word for word in textblob_doc.words.lower().singularize() if d.check(word)])  # need to elim this d.check(word) part in order to get proper names in output
        #doc_lc_sing_real = ' '.join([word for word in textblob_doc.words.lower().singularize()])  # need to elim this d.check(word) part in order to get proper names in output
        textblob_doc = TextBlob(doc_lc_sing_real)
        only_part_of_speech = ' '.join([word_tag[0] for word_tag in textblob_doc.tags if word_tag[1] == 'VB' or word_tag[1] == 'VBD' or word_tag[1] == 'VBG' or word_tag[1] == 'VBN' or word_tag[1] == 'VBP' or word_tag[1] == 'VBZ' or word_tag[1] == 'NN' or word_tag[1] == 'NNS'])
        only_VB_NN_combined_sent_w_emo_docs.append(only_part_of_speech)
    return only_VB_NN_combined_sent_w_emo_docs


def tfidf_vectorize(all_emo_variations, combined_sent_around_emo_docs):
    # add all emotion words to stoplist
    my_words = set(all_emo_variations + ['felt', 'feel', 'feeling', 'feels'])
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    #vectorizer = TfidfVectorizer(stop_words="english")  #orig text just using "english" stopwords
    vectorizer = TfidfVectorizer(stop_words=set(my_stop_words))  # add all emo words in dict as stop words
    words_around_emo_vectors = vectorizer.fit_transform(combined_sent_around_emo_docs)  #this is a list of strings.
    #vectorizer.get_feature_names()[5]  # gives names of words
    return words_around_emo_vectors, vectorizer


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


# gives tfidfs for each emo labeled as either belonging to corpus1 or corpus2
def alt_master_corpus_to_emo_to_tfidf_term_dict(corpus1, corpus2, root_to_variations_dict):
    all_emo_variations = create_all_emo_variations_list(root_to_variations_dict)
    combined_sent_around_emo_docs12, emo_list12 = create_sentences_w_emo_and_create_emo_list_two_corp(corpus1, corpus2)
    #if want to do just nouns or verbs, de-comment one of these below
    #combined_sent_around_emo_docs12 = only_NN_combined_sent_with_emo_docs(combined_sent_around_emo_docs12)
    #combined_sent_around_emo_docs12 = only_VB_combined_sent_with_emo_docs(combined_sent_around_emo_docs12)
    combined_sent_around_emo_docs12 = only_VB_NN_combined_sent_with_emo_docs(combined_sent_around_emo_docs12)
    #select one of the below two options for either tf-idf or just tf:
    words_around_emo_vectors, vectorizer = tfidf_vectorize(all_emo_variations, combined_sent_around_emo_docs12)
    #words_around_emo_vectors, vectorizer = tf_vectorize(all_emo_variations, combined_sent_around_emo_docs12)
    root_to_tfidf_terms_dict = create_emo_to_tfidf__term_dict(vectorizer, combined_sent_around_emo_docs12, words_around_emo_vectors, emo_list12)
    # write these dicts again to the emotion_flask_app folder so can be used by app
    # with open('root_to_tfidf_terms_dict.pkl', 'w') as picklefile:
    #     pickle.dump(root_to_tfidf_terms_dict, picklefile)
    return root_to_tfidf_terms_dict



# these are for use after user inputs an emotion. they... print results at the moment.
# where to save info to use in html?
def give_assoc_words_and_sentences_one_corpora(root_to_tfidf_terms_dict_combo_corpora, root_to_sentences_dict_corpX, term_corpusX, emo):
    words = []
    docs_w_words = []
    init_results = root_to_tfidf_terms_dict_combo_corpora[term_corpusX][0][:5]
    results = [tuple for tuple in init_results if tuple[1] > .1]  #this sorts by the 2nd item in the tuple, the tf-idf score
    for result in results:
        words.append(result)
        #print len(root_to_sentences_dict_corpX[emo])
        for i in range(len(root_to_sentences_dict_corpX[emo])):
            if result[0] in set(TextBlob(root_to_sentences_dict_corpX[emo][i]).words.lower().singularize()):
                doc_w_word = root_to_sentences_dict_corpX[emo][i]
                docs_w_words.append(doc_w_word)
                break
    return words, docs_w_words

def print_results_from_both_corpora(root_to_tfidf_terms_dict_combo_corpora, root_to_sentences_dict_corp1, root_to_sentences_dict_corp2, emo):
    term_corpus1 = emo + '1'
    term_corpus2 = emo + '2'
    words_1, docs_w_words_1 = give_assoc_words_and_sentences_one_corpora(root_to_tfidf_terms_dict_combo_corpora, root_to_sentences_dict_corp1, term_corpus1, emo)
    words_2, docs_w_words_2 = give_assoc_words_and_sentences_one_corpora(root_to_tfidf_terms_dict_combo_corpora, root_to_sentences_dict_corp2, term_corpus2, emo)
    return words_1, docs_w_words_1, words_2, docs_w_words_2
