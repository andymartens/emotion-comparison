# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:49:44 2015

@author: charlesmartens
"""

#GET GOOD LIST IN TEXT FILE OF STORM AND CLORE AND OTHER WORDS THAT I CAN 
#ALWAYS GO BACK TO. THEN CAN COMPARE AND WORK WITH
#...


#this pattern module has wordnet in it. and has kind of replaced
#just using wordnet alone
from pattern import en
from pattern.en import conjugate, lemma, lexeme
from pattern.en import tenses, PAST, PL
from pattern.en import wordnet as wn
import pandas as pd
import enchant  #why not working??
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import pickle
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns


#from nltk.corpus import wordnet as wn
#from nltk.stem.wordnet import WordNetLemmatizer

#so, want to put the root word here and then get variations?
#this is good. can put any word in and it'll give me variations:
lexeme('worship')  

#this will take any variation and give me root:
lemma('irritate')  #this could be used to change words back to the root in the corpus

#what do i need? PLAN.
#a dict that goes from the variations on a word to the root (or shortest form)
#should i use lexeme to build out the variations. ok. 
#then do a set to make unique
#then go over myself and take out bogus words and non-emo first words
#create dict of root to variations. did using stemmer before. could also
#use lemma? but that might not work well? use both? and then do by hand


#get fletcher emo word list to compare:
#datafile = open('emotion_words_fletcher.txt', 'r')
#data = []
#for row in datafile:
#    data.append(row.strip())
#emo_words_fletcher = data


#get storm emo words
datafile = open('emotion_words_storm.txt', 'r')
data = []
for row in datafile:
    data.append(row.strip().split(','))

print data[0]
len(data[0])

words_storm = [word.replace('\r', '').strip().lower() for word in data[0]]

words_storm_2 = []
for word in words_storm:
    words_in_line = word.split(' ')
    words_storm_2 = words_storm_2 + words_in_line

words_storm_2 = sorted(words_storm_2)

outputFile = open('emotion_words_storm_2.txt', 'w')  
for word in words_storm_2: 
    outputFile.write(word+'\n')
outputFile.close()    


#get clore emo words
datafile = open('emotion_words_clore.txt', 'r')
data = []
contents = datafile.read()
data.append(contents.strip().split('\r'))
datafile.close()

words_clore = []
for word in data[0]:
    words_in_line = word.split(' ')
    words_clore = words_clore + words_in_line

len(words_clore)

words_clore_2 = sorted(words_clore)

outputFile = open('emotion_words_clore_2.txt', 'w')  
for word in words_clore_2: 
    outputFile.write(word+'\n')
outputFile.close()    


#put storm and clore words together
all_words = words_clore + words_storm_2 + emo_words_fletcher
len(set(all_words))

all_words_sorted_unique = sorted(list(set(all_words)))

all_words_sorted_unique = all_words_sorted_unique[1:]
all_words_sorted_unique[:50]
len(all_words_sorted_unique)


#get words to text doc 
outputFile = open('all_emo_words_1.txt', 'w')  #creates a file object called outputFile. It also 
for word in all_words_sorted_unique: 
    outputFile.write(word+'\n')
outputFile.close()    


#get words back:
datafile = open('all_emo_words_1.txt', 'r')
data = []
for row in datafile:
    data.append(row.strip())

all_emo_words_2 = data


#test_list = ['fear', 'abandon', 'awe', 'shame', 'guilt', 'joy', 'happy', 'horrifying']
#variations = add_lexeme_variations(test_list)
#variations_real_words = eliminate_nonwords(variations)  #combine with above step/function
#variations_w_adverbs = add_adverb_variations(variations_real_words)


#add variations on emo words to the list
def add_lexeme_variations(word_list):
    lexemes_list = []
    for emo_word in word_list:
        lexemes = lexeme(emo_word)
        lexemes_list = lexemes_list + lexemes
    all_words = lexemes_list + word_list
    return sorted(list(set(all_words)))

#lexemes_list = []
#for emo_word in all_emo_words_2:
#    lexemes = lexeme(emo_word)
#    lexemes_list = lexemes_list + lexemes
#
#len(lexemes_list)
#
#all_emo_words_3 = all_emo_words_2 + lexemes_list
#len(all_emo_words_3)
#
#all_emo_words_4 = sorted(list(set(all_emo_words_3)))
#len(all_emo_words_4)
#all_emo_words_4[:100]

#get adverbs here. just add ly to all words. and then go through and remove words

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
    
#adverbs = []
#for word in all_emo_words_4:
#    adverbs.append(word+'ly')
#    
#len(adverbs)
#adverbs[20:70]

#all_emo_words_5 = all_emo_words_4 + adverbs 
#all_emo_words_5 = sorted(all_emo_words_5)
#len(all_emo_words_5)
#all_emo_words_5[:20]


#get words to text doc 
outputFile = open('all_emo_words_3.txt', 'w')
for word in all_emo_words_5: 
    outputFile.write(word+'\n')
outputFile.close()    


#get words back:
datafile = open('all_emo_words_3.txt', 'r')
data = []
for row in datafile:
    data.append(row.strip())

#all_emo_words_6 = data
#len(all_emo_words_6)


#remove words that aren't real from all_emo_words_6.

#d = enchant.Dict("en_US")
#d.check("yearnly")
#
#all_emo_words_7 = []
#for word in all_emo_words_6:
#    if d.check(word):
#        all_emo_words_7.append(word)
#
#len(all_emo_words_7)
#all_emo_words_7[100:200]
#

#words_not_in_new_list = []
#for word in emo_words_fletcher:
#    if word not in all_emo_words_7:
#        words_not_in_new_list.append(word)
#
#len(words_not_in_new_list)
#words_not_in_new_list[:50]


#function that takes other prior functions and adds all variations to 
#a words list, and makes sure they're real words. 
#i.e., lexeme variations and adverb variations and then eliminates non-words:
def add_all_word_variations(word_list):
    new_word_list = add_lexeme_variations(word_list)
    new_word_list = add_adverb_variations(new_word_list)
    return new_word_list


#get into text file again and go through manually
#outputFile = open('all_emo_words_4.txt', 'w')  #creates a file object called outputFile. It also 
#for word in all_emo_words_7: 
#    outputFile.write(word+'\n')
#outputFile.close()    

#OK, SAVE COMPLETE LIST TO all_emo_words_4.txt BUT IT'S HUGE. TOO BIG
#WORK LATER ON CUTTING DOWN SOMEHOW TO WORDS TAHT ARE MORE CLEARLY EMOTION
#HOW TO DO THIS? OTHER THAN WITH MY OWN JUDGEMENT? COULD USE CLORE LIST
#OR USE STORM LIST SEPARATELY? IS EITHER ONE MORE STANDARD EMOTION?

#LOOK AGAIN AT WHAT WORDS WEREN'T CAUGHT BY STORM AND CLORE. ARE THEY IMPORTANT?

#LEFT OFF, LEFT OFF
#AT THIS POINT, NEED TO GO THROUGH all_emo_words_4.txt AND TAKE OUT BAD WORDS
#(E.G., WIND) AND THEN GO THROUGH CODE FROM THIS POINT ON.
#I.E., IMPORT all_emo_words_4.txt BACK FROM THE TEXT FILE AFTER I MESS W IT.
#AND THEN DO BELOW STUFF


#NEXT STEP: dict that goes from the variations on a word to the root (or shortest form)
# make dict with stem the key and the actual words values. 
#then make dict of stemmed word to one of the words (shortest word?)


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


#stemmer = SnowballStemmer('english')
#emotion_words_complete_dict = defaultdict(list)
#
#for word in all_emo_words_7:
#    #word = TextBlob(word)
#    stemmed_word1 = stemmer.stem(word)
#    for word2 in all_emo_words_7:
#        #word2 = TextBlob(word2)
#        stemmed_word2 = stemmer.stem(word2)
#        if stemmed_word1 == stemmed_word2:
#            emotion_words_complete_dict[stemmed_word1].append(word2)
#
#len(emotion_words_complete_dict.keys())


#make ea list of values a set:

#for key in emotion_words_complete_dict:
#    emotion_words_complete_dict[key] = list(set(emotion_words_complete_dict[key]))
#
#
##test_dict = {}
#for i, key in enumerate(emotion_words_complete_dict.keys()):
#    if i < 10:        
#        print key, emotion_words_complete_dict[key]
#        #test_dict[key] = emotion_words_complete_dict[key]


#test_dict = {'uncomfort': ['uncomfortable'], 'consider': ['consideration', 
#'considerate', 'considerations', 'considerately'], 'dispirit': ['dispirits', 
#'dispiriting', 'dispirited', 'dispirit']}


#replace key with first or shortest word in values

#for key in emotion_words_complete_dict.keys():
#    new_key = emotion_words_complete_dict[key][0]
#    for word in emotion_words_complete_dict[key]:
#        if word[-2:] == 'ed':
#            new_key = word  
#    emotion_words_complete_dict[new_key] = emotion_words_complete_dict.pop(key)


#pickle this list so i don't have to go through above steps to get the emo_dict

#to pickle:
with open('emo_words_dict.pkl', 'w') as picklefile:
    pickle.dump(emotion_words_complete_dict, picklefile)
    
#to retrieve:
with open('emo_words_dict.pkl', 'r') as picklefile:
    emotion_words_complete_dict_2 = pickle.load(picklefile)


#write complete emo words to file so can view
#outputFile = open('all_emo_words_5.txt', 'w')  #creates a file object called outputFile. 
#for key in emotion_words_complete_dict_2.keys():
#    for word in emotion_words_complete_dict_2[key]: 
#        outputFile.write(word+'\n')
#outputFile.close()    


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


storm_words_dict = open_emo_words_and_to_dict('emotion_words_storm_2.txt')
clore_words_dict = open_emo_words_and_to_dict('emotion_words_clore_2.txt')


def open_emo_words_and_to_list(text_file):
    #get words back:
    datafile = open(text_file, 'r')
    data = []
    for row in datafile:
        data.append(row.strip())
    emotion_words = data
    
    emotion_words_variations = add_all_word_variations(emotion_words)
    return emotion_words_variations

storm_words = open_emo_words_and_to_list('emotion_words_storm_2.txt')
clore_words = open_emo_words_and_to_list('emotion_words_clore_2.txt')
fletcher_words = open_emo_words_and_to_list('emotion_words_fletcher.txt')


# don't think this matters. skip:
#sent words to txt file and back to get rid of unicode
def send_words_to_txt_and_back(word_list):
    outputFile = open('temp_file.txt', 'w')  
    for word in word_list: 
        outputFile.write(word+'\n')
    outputFile.close()    
    datafile = open('temp_file.txt', 'r')
    data = []
    for row in datafile:
        data.append(row.strip())
    datafile.close()
    return data

storm_words = send_words_to_txt_and_back(storm_words)
clore_words = send_words_to_txt_and_back(clore_words)
fletcher_words = send_words_to_txt_and_back(fletcher_words)


def word_list_to_text_file(word_list, text_file_name):
    outputFile = open(text_file_name, 'w')  
    for word in word_list: 
        outputFile.write(word+'\n')
    outputFile.close()    

word_list_to_text_file(storm_words, 'emotion_words_storm_3.txt')
word_list_to_text_file(clore_words, 'emotion_words_clore_3.txt')
word_list_to_text_file(fletcher_words, 'emotion_words_fletcher_3.txt')


#comare two word lists:
def show_words_not_in_list2(list_1, list_2):
    different_words = []
    for word in list_1:
        if word not in list_2:
            different_words.append(word)
    return different_words


storm_vs_clore = show_words_not_in_list2(storm_words, clore_words)
len(storm_vs_clore)

clore_vs_storm = show_words_not_in_list2(clore_words, storm_words)
len(clore_vs_storm)


emo_words_to_add = ['guilty', 'hate', 'hated', 'hates', 'hating', 'sad', 
                    'sadden', 'saddened', 'saddening', 'saddens', 'sadly', 
                    'scary', 'weep', 'weeping', 'weeps', 'weepy', 'wept',
                    'ashamed' 'shameful', 'shyness'] 


clore_and_storm = sorted(list(set(storm_words + clore_words + emo_words_to_add)))
len(clore_and_storm)
clore_and_storm[100:200]

clore_storm_vs_fletcher = show_words_not_in_list2(fletcher_words, clore_and_storm)
len(clore_storm_vs_fletcher)
clore_storm_vs_fletcher[:100]

word_list_to_text_file(clore_and_storm, 'clore_and_storm_Mar19.txt')

clore_and_storm_Mar19_dict = create_dict_of_word_variations(clore_and_storm)

#to create pickle
with open('clore_and_storm_words_Mar19_dict.pkl', 'w') as picklefile:
    pickle.dump(clore_and_storm_Mar19_dict, picklefile)


######################################################################
#Use this dict for now as the emotion dictionary. tweak later.
#add words from list to add below. and take out lots of words
#whose primary meaning is non-emo.
#to retrieve pickle:
with open('clore_and_storm_words_Mar19_dict.pkl', 'r') as picklefile:
    clore_and_storm_Mar19_dict = pickle.load(picklefile)
######################################################################


#now that have dict of emo words...
#get dream and waking corpuses ready to work with.
#but final code for this on code_emo_app.py can ignore here:

#connect to database:
client = pymongo.MongoClient()  #create a MongoClient to the running mongod instance:
db = client.dreams  
dream_wake_collection = db.dream_wake 

#get and clean waking reports
cursor_wake = dream_wake_collection.find( {'dream_wake': 'Waking'}, {'text':1, '_id':0})
waking_corpus = [report['text'] for report in cursor_wake]
waking_corpus_clean = [report for report in waking_corpus if len(report) > 150]  

#get rid of duplicate waking reports:
waking_corpus_clean = [report for report in waking_corpus_clean if report != waking_corpus_clean[0] 
                           and report != waking_corpus_clean[1] and report != waking_corpus_clean[2] 
                           and report != waking_corpus_clean[10] and report != waking_corpus_clean[12]
                           and report != waking_corpus_clean[17] and report != waking_corpus_clean[19]
                           and report != waking_corpus_clean[69] and report != waking_corpus_clean[70]]

len(waking_corpus_clean)

with open('waking_corpus_clean.pkl', 'w') as picklefile:
    pickle.dump(waking_corpus_clean, picklefile)

################################################################
#to get waking reports corpus:
with open('waking_corpus_clean.pkl', 'r') as picklefile:
    waking_corpus_clean_2 = pickle.load(picklefile)
################################################################
len(waking_corpus_clean_2[:3])

with open('corpus1_test.txt', 'w') as my_file:
    my_file.write(waking_corpus_clean_2[0])

#get and clean dream reports
cursor_dreams = dream_wake_collection.find( {'dream_wake': 'Dream'}, {'text':1, '_id':0})
dream_corpus = [dream['text'] for dream in cursor_dreams]
dream_corpus_clean = [dream for dream in dream_corpus if len(dream) > 150]  

#get rid of duplicate dreams:
dream_corpus_clean = [dream for dream in dream_corpus_clean if dream != dream_corpus_clean[0] 
                           and dream != dream_corpus_clean[1] and dream != dream_corpus_clean[9] 
                           and dream != dream_corpus_clean[14] and dream != dream_corpus_clean[15]
                           and dream != dream_corpus_clean[16] and dream != dream_corpus_clean[17]
                           and dream != dream_corpus_clean[68] and dream != dream_corpus_clean[69]
                           and dream != dream_corpus_clean[183]]

with open('dream_corpus_clean.pkl', 'w') as picklefile:
    pickle.dump(dream_corpus_clean, picklefile)

################################################################
#to get dream reports corpus:
with open('dream_corpus_clean.pkl', 'r') as picklefile:
    dream_corpus_clean_2 = pickle.load(picklefile)
################################################################


with open('corpus1_test.txt', 'w') as my_file:
    for doc in dream_corpus_clean_2:        
        my_file.write('"""' + doc + '"""' + ', ')

with open('corpus2_test.txt', 'w') as my_file:
    for doc in waking_corpus_clean_2:        
        my_file.write('"""' + doc + '"""' + ', ')


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
corpuses_to_plot(dream_corpus_clean_2, waking_corpus_clean_2, 'Dreams', 'Real-life', clore_and_storm_Mar19_dict)
######################################################################################


#maybe to add? 
supplemental_list = ['unworthy', 
'worthy', 'unsettle', 'unsettled',
 u'unsettles',
 u'unsettling',
 u'stress',
 u'stressed',
 u'stresses',
 u'stressing',
 u'sulk',
 u'sulked',
 u'sulking',
 u'sulks',
 'sullen',
 'sullenly',
 'tearful',
 'tearfully',
 u'tense',
 u'tensely',
 u'threat',
 u'threats',
 u'repulse',
 u'repulsed',
 u'repulses',
 u'repulsing',
 u'resent',
 u'resented',
 'resentful',
 'resentfully',
 u'resenting',
 u'resents',
 u'seethed',
 u'seethes',
 u'seething',
 'somber',
 'somberly',
 'sorrowful',
 'sorrowfully',
 u'obsess',
 u'obsessed',
 u'obsesses',
 u'obsessing',
 'obsessive',
 'obsessively',
 'obsessives',
 u'overwhelm',
 u'overwhelmed',
 u'overwhelming',
 u'overwhelms',
 'panicky',
 'paranoia',
 'paranoid',
 'paranoids',
 'powerless',
 'powerlessly',
 'regretful',
 'regretfully',
 'remorseful',
 'remorsefully'
 'miserable',
 'happiness',
 'helplessness',
 'hothead',
 'hotheaded',
 'hotheadedly',
 'hotheads',
 'humility',
 'hurtful',
 'hurtfully',
 u'idolize',
 u'idolized',
 u'idolizes',
 u'idolizing',
 u'infuriate',
 u'infuriated',
 u'infuriates',
 u'infuriating',
 u'infuriatingly',
 'jealous',
 'jealously',
 'judgmental',
 'judgmentally'
 u'enrage',
 u'enraged',
 u'enrages',
 u'enraging',
 u'exhilarate',
 u'exhilarated',
 u'exhilarates',
 u'exhilarating',
 'exhilaration',
 u'fluster',
 u'flustered',
 u'flustering',
 u'flusters',
 u'fret',
 'fretful',
 'fretfully',
 u'frets',
 u'fretted',
 u'fretting',
 u'fume',
 u'fumed',
 u'fuming',
 'genial',
 'genially',
 u'grieve',
 u'grieved',
 u'grieves',
 u'grieving',
 'grim',
 'grimly',
 u'grouch',
 'dismal',
 'dismally',
 u'disorientate',
 u'disoriented',
 u'disorienting',
 u'disorients',
 u'disquiet',
 u'disquieted',
 u'disquieting',
 u'disquiets',
 'disrespectful',
 'disrespectfully',
 'dopey',
 'downcast',
 'ebullient',
 'ebulliently',
 u'bummed',
 'bummer',
 'bummers',
 u'chagrin',
 u'chagrins',
 u'cherish',
 u'cherished',
 u'cherishes',
 u'cherishing',
 'compulsive',
 'compulsively',
 'aggravation',
 'aggravations',
 u'agonize',
 u'agonized',
 u'agonizes',
 u'agonizing',
 u'agonizingly',
 'ambivalent',
 'ambivalently',
 u'antagonize',
 u'antagonized',
 u'antagonizes',
 u'antagonizing',
 u'appall',
 u'appalled',
 u'appalling',
 u'appallingly',
 'appreciative',
 'appreciatively',
 'attentive',
 'attentively',
 'awkward',
 'awkwardly',
 'awkwardness',
 u'befuddle',
 u'befuddled',
 u'befuddles',
 u'befuddling',
 u'belittle',
 u'belittled',
 u'belittles',
 u'belittling',
 'belligerent',
 'belligerently',
 'belligerents',
 'bleak',
 'bleakly',
 'bleakness']



# get two dictionaries ready:
# root_to_variations_dict. this has each emo word and values are list of variations
# and the coresponding_root_to_ratings_dict. this has each emo root/key in 
# root_to_variations_dict and the valence, arousal, and intensity ratings for 
# root word as the values

# first, create big list of emo words that includes all variations of words:
clore_and_storm_Mar19_dict

big_list = []
for key, values in clore_and_storm_Mar19_dict.items():
    big_list += [key]
    big_list += values

len(big_list)
big_list[:20]

test_set = set(big_list)
len(test_set)

len(supplemental_list)

big_list = big_list + supplemental_list  #all words so far

def word_list_to_text_file(word_list, text_file_name):
    outputFile = open(text_file_name, 'w')  
    for word in word_list: 
        outputFile.write(word+'\n')
    outputFile.close()    

word_list_to_text_file(big_list, 'words_list_to_match_shaver_storm.txt')

# get 'words_to_add' and combine with big_list
df_words_to_add = pd.read_excel('words_to_add.xlsx') 
words_to_add = df_words_to_add.values
words_to_add = [word[0] for word in words_to_add]

biggest_list = big_list + words_to_add
biggest_list = sorted(list(set(biggest_list)))
biggest_list[:25]
len(biggest_list)


# create vairation on biggest_list 
# turn into dict
# pair down in excel? could get dict into df and then into excel
    # w each line consisting of key root word and value variations
# then get back into a df and then a dict

biggest_list_variations = add_all_word_variations(biggest_list)
biggest_list_dict = create_dict_of_word_variations(biggest_list_variations)

def dict_to_csv(word_dict, csv_file_name):
    outputFile = open(csv_file_name, 'w')  
    for key in word_dict.keys(): 
        outputFile.write(key + ',') 
        for value in word_dict[key]:
            outputFile.write(value + ',') 
        outputFile.write('\n') 
    outputFile.close()    

dict_to_csv(biggest_list_dict, 'biggest_list_from_dict.csv')


# take biggest_list_from_dict_paired.csv  -  and put into dict.
    # how? put into df?
    # or direct into dict with open function -- maybe better?
    # then i'll have my foundation dictionary.
    # will then need a dict from those keys to the valence/intensity score
    # then -- make graphs verticle and color bars according to valence/intensity

import csv

# takes csv with root word in first col and variations in subseq cols
# and converts it to a dict with key as root words and variations as values
def convert_csv_variations_to_dict(csv_file):
    root_to_variations_dict = {}
    with open(csv_file, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            root_to_variations_dict[row[0]] = [word for word in row[1:] if word != '']
    return root_to_variations_dict


# IN FUTURE, TO ALTER CORE EMO WORD DICT, START W:
# 1. OPEN 'biggest_list_from_dict_paired.csv' IN EXCEL. THE FIRST COL IS THE 
# ROOT WORD, THE SUBSQ COLS ARE VARIATIONS. HERE I CAN MESS WITH BY HAND, ELIM 
# WORDS AND VARIATIONS, ADD WORDS AND VARIATIONS, ETC.
# 2. DO THE FOLLOWING TO CONVERT THIS .CSV FILE TO A DICT WITH ROOT WORD AS
# THE KEY AND VARIATIONS OF WORD AS THE VALUES
root_to_variations_dict = convert_csv_variations_to_dict('biggest_list_from_dict_paired.csv')
root_to_variations_dict.pop('root')

len(root_to_variations_dict.keys())
root_to_variations_dict['happy']

#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# get list of emo words with valence and arousal ratings (Warriner et al csv file)
# and turn into dict with word as key and valence and arousal and product of the
# two numbers as values

# 3. CONVERT THE LIST OF WARRINER WORDS AND RATINGS TO A DICT
def convert_csv_word_ratings_to_dict(csv_file):
    word_to_ratings_dict = {}
    with open(csv_file, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for row in reader:  
            valence = float(row[2]) - 5.0
            arousal = float(row[5])
            intensity = valence * arousal
            word_to_ratings_dict[row[1]] = [valence, arousal, intensity]
    return word_to_ratings_dict

root_to_ratings_dict = convert_csv_word_ratings_to_dict('Ratings_Warriner_et_al.csv')

# need to create more keys for words that are in root_to_variations dict but
# not in root_to_ratings dict. so:
# if word in root_to_variations is not a key in root_to_ratings, then create
# that key in root_to_ratings and make the values... the values for the first
# variation on root_to_variations. if 


# next: 
# pickle both dicts
# in code_emo_app.py file, unpickle dicts
# run graph once with this new dict
# alter graph so it's verticle
# make shade or hue or transparency of bars in graph represent my product-intensity score


with open('root_to_variations_dict.pkl', 'w') as picklefile:
    pickle.dump(root_to_variations_dict, picklefile)

with open('root_to_ratings_dict.pkl', 'w') as picklefile:
    pickle.dump(root_to_ratings_dict, picklefile)


# ok, problem is that not in this dictionary!? becasue not all words in the
# root to variations dicdt are in the big root to ratings dict!
# but at least one of the words will be. so if can't find key in root_to_ratings dict
# then go to the first value word in variation dict and search for that, and so on
# code: if key from root_to_variations_dict in list of keys from root_to_ratings_dict
# then all's good. elif second variation from root_to_variations_dict is in 
# root_to_ratings_dict, then create new key in root_to_ratings_dict with the key
# from the root for root_to_variations_dict and give it the ratings of the second
# variation.  EASIER AND MORE ROBUST WAY TO DO THIS?
#test_list = ['fear', 'bummed', 'cocky', 'cheeky', 'challenged']
#testing_root_to_variations_dict = {}
#for emo in test_list:
#    testing_root_to_variations_dict[emo] = root_to_variations_dict[emo]


# creating a new root to ratings dict that will correspond to the words 
# in the root to variations dict.
# this works gret. make into functino and document this so can repeat it
# when/if refine the emo to variations list.

# 4. CREATE A NEW ROOT TO RATINGS DICT WHOSE ROOTS ARE THE SAME WORDS AS THE ROOTS
# IN THE ROOTS_TO_VARIATIONS_DICT AND WITH VALENCE, AROUSAL, AND INTENSITY RATINGS
# THAT ARE MEANS OF RATINGS OF ALL VARIATIONS ON EACH ROOT (INCLUDING THE ROOT)
def create_corresponding_root_to_ratings_dict(root_to_variations_dict, root_to_ratings_dict):
    corresponding_root_to_ratings_dict = {}
    for key in root_to_variations_dict.keys():
        #ratings_dict = defaultdict(list)   
        ratings_dict = {'valence': [], 'arousal': [], 'intensity': []}
        for variation in root_to_variations_dict[key]:
            if variation in root_to_ratings_dict:
                ratings_dict['valence'].append(root_to_ratings_dict[variation][0])
                ratings_dict['arousal'].append(root_to_ratings_dict[variation][1])
                ratings_dict['intensity'].append(root_to_ratings_dict[variation][2])
            else:
                ratings_dict['valence'].append(np.nan)
                ratings_dict['arousal'].append(np.nan)
                ratings_dict['intensity'].append(np.nan)
        mean_valence = round(np.nanmean(ratings_dict['valence']), 4)
        mean_arousal = round(np.nanmean(ratings_dict['arousal']), 4)
        mean_intensity = round(np.nanmean(ratings_dict['intensity']), 4)
        corresponding_root_to_ratings_dict[key] = [mean_valence, mean_arousal, mean_intensity]
    return corresponding_root_to_ratings_dict

corresponding_root_to_ratings_dict = create_corresponding_root_to_ratings_dict(root_to_variations_dict, root_to_ratings_dict)


len(corresponding_root_to_ratings_dict)
len(root_to_variations_dict)

# 5. PICKLE THIS NEW corresponding_root_to_ratings_dict (AS WELL AS THE root_to_variations_dict)
# TO BE USED WITH THE APP TO MAKE GRAPHS, ETC.
with open('corresponding_root_to_ratings_dict.pkl', 'w') as picklefile:
    pickle.dump(corresponding_root_to_ratings_dict, picklefile)










