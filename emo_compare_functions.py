# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:08:10 2015

@author: charlesmartens
"""

from pattern import en
from pattern.en import conjugate, lemma, lexeme
from pattern.en import tenses, PAST, PL
from pattern.en import wordnet as wn
import pandas as pd
import enchant  #why not working??
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import pickle



#add variations on emo words to the list
def add_lexeme_variations(word_list):
    lexemes_list = []
    for emo_word in word_list:
        lexemes = lexeme(emo_word)
        lexemes_list = lexemes_list + lexemes
    all_words = lexemes_list + word_list
    return sorted(list(set(all_words)))


#helper function for adverbs function: gets rid of words that aren't real words
def eliminate_nonwords(word_list):    
    d = enchant.Dict("en_US")
    all_words = []
    for word in word_list:
        if d.check(word):
            all_words.append(word)
    return all_words


def add_adverb_variations(word_list):
    adverbs = []
    for word in word_list:
        adverbs.append(word+'ly')
    all_words = word_list + adverbs
    all_words = sorted(list(set(all_words)))
    all_words = eliminate_nonwords(all_words)
    return all_words


#function that takes other prior functions and adds all variations to 
#a words list, i.e., lexeme variations and adverb variations:
def add_all_word_variations(word_list):
    new_word_list = add_lexeme_variations(word_list)
    new_word_list = add_adverb_variations(new_word_list)
    return new_word_list


#helper function for create_dict_of_word_variations: makes values unique:
def make_dict_values_unique(word_to_variations_dict):
    for key in word_to_variations_dict:
        word_to_variations_dict[key] = list(set(word_to_variations_dict[key]))
    return word_to_variations_dict


#helper to create_dict_of_word_variations: changes key  back to real word:
def replace_key_with_real_word(word_to_variations_dict):
    for key in word_to_variations_dict.keys():
        new_key = word_to_variations_dict[key][0]
        for word in word_to_variations_dict[key]:
            if word[-2:] == 'ed':
                new_key = word  
        word_to_variations_dict[new_key] = word_to_variations_dict.pop(key)
    return word_to_variations_dict


def create_dict_of_word_variations(word_list):
    stemmer = SnowballStemmer('english')
    emotion_words_complete_dict = defaultdict(list)
    for word in word_list:
        stemmed_word1 = stemmer.stem(word)
        for word2 in word_list:
            stemmed_word2 = stemmer.stem(word2)
            if stemmed_word1 == stemmed_word2:
                emotion_words_complete_dict[stemmed_word1].append(word2)
    emotion_words_complete_dict = make_dict_values_unique(emotion_words_complete_dict)
    emotion_words_complete_dict = replace_key_with_real_word(emotion_words_complete_dict)
    return emotion_words_complete_dict


def open_emo_words_and_to_dict(text_file):
    #get words back:
    datafile = open(text_file, 'r')
    data = []
    for row in datafile:
        data.append(row.strip())
    emotion_words = data
    
    emotion_words_variations = add_all_word_variations(emotion_words)
    emotion_words_dict = create_dict_of_word_variations(emotion_words_variations)
    return emotion_words_dict


def open_emo_words_and_to_list(text_file):
    #get words back:
    datafile = open(text_file, 'r')
    data = []
    for row in datafile:
        data.append(row.strip())
    emotion_words = data
    
    emotion_words_variations = add_all_word_variations(emotion_words)
    return emotion_words_variations


#Kind of start here: take what are basically cleaned up lists from the
#articles and this function makes variations, etc. so now returns
#fuller flexhed out list with lexemes and adverb versions.
storm_words = open_emo_words_and_to_list('emotion_words_storm_2.txt')
clore_words = open_emo_words_and_to_list('emotion_words_clore_2.txt')
fletcher_words = open_emo_words_and_to_list('emotion_words_fletcher.txt')


#some words that were missed with above function to flesh out the words
#either obvious and needed to be in there, like sad. or just that function
#didn't work well enough to create these words. e.g., had guilt but
#function didn't know enough to create guily.
emo_words_to_add = ['guilty', 'hate', 'hated', 'hates', 'hating', 'sad', 
                    'sadden', 'saddened', 'saddening', 'saddens', 'sadly', 
                    'scary', 'weep', 'weeping', 'weeps', 'weepy', 'wept',
                    'ashamed' 'shameful', 'shyness'] 


clore_and_storm = sorted(list(set(storm_words + clore_words + emo_words_to_add)))


def word_list_to_text_file(word_list, text_file_name):
    outputFile = open(text_file_name, 'w')  
    for word in word_list: 
        outputFile.write(word+'\n')
    outputFile.close()    


word_list_to_text_file(clore_and_storm, 'clore_and_storm_Mar19.txt')


clore_and_storm_Mar19_dict = create_dict_of_word_variations(clore_and_storm)
