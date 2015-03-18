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
