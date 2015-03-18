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
datafile = open('emotion_words_fletcher.txt', 'r')
data = []
for row in datafile:
    data.append(row.strip())
emo_words_fletcher = data


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


#in text doc, go through words and remove words w non-emo primary meanings

#get words back:
datafile = open('all_emo_words_1.txt', 'r')
data = []
for row in datafile:
    data.append(row.strip())

all_emo_words_2 = data


#add variations on emo words to the list
def add_lexeme_variations(word_list):
    lexemes_list = []
    for emo_word in word_list:
        lexemes = lexeme(emo_word)
        lexemes_list = lexemes_list + lexemes
    all_words = lexemes_list + word_list
    return sorted(list(set(all_words)))

lexemes_list = []
for emo_word in all_emo_words_2:
    lexemes = lexeme(emo_word)
    lexemes_list = lexemes_list + lexemes

len(lexemes_list)

all_emo_words_3 = all_emo_words_2 + lexemes_list
len(all_emo_words_3)

all_emo_words_4 = sorted(list(set(all_emo_words_3)))
len(all_emo_words_4)
all_emo_words_4[:100]

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
    
adverbs = []
for word in all_emo_words_4:
    adverbs.append(word+'ly')
    
len(adverbs)
adverbs[20:70]

all_emo_words_5 = all_emo_words_4 + adverbs 
all_emo_words_5 = sorted(all_emo_words_5)
len(all_emo_words_5)
all_emo_words_5[:20]

#get words to text doc 
outputFile = open('all_emo_words_3.txt', 'w')  #creates a file object called outputFile. It also 
for word in all_emo_words_5: 
    outputFile.write(word+'\n')
outputFile.close()    


#get words back:
datafile = open('all_emo_words_3.txt', 'r')
data = []
for row in datafile:
    data.append(row.strip())

all_emo_words_6 = data
len(all_emo_words_6)


#remove words that aren't real from all_emo_words_6.

d = enchant.Dict("en_US")
d.check("yearnly")

all_emo_words_7 = []
for word in all_emo_words_6:
    if d.check(word):
        all_emo_words_7.append(word)

len(all_emo_words_7)
all_emo_words_7[100:200]


#words_not_in_new_list = []
#for word in emo_words_fletcher:
#    if word not in all_emo_words_7:
#        words_not_in_new_list.append(word)
#
#len(words_not_in_new_list)
#words_not_in_new_list[:50]


#function that takes other prior functions and adds all variations to 
#a words list, i.e., lexeme variations and adverb variations:
def add_all_word_variations(word_list):
    new_word_list = add_lexeme_variations(word_list)
    new_word_list = add_adverb_variations(new_word_list)
    return new_word_list




#get into text file again and go through manually
outputFile = open('all_emo_words_4.txt', 'w')  #creates a file object called outputFile. It also 
for word in all_emo_words_7: 
    outputFile.write(word+'\n')
outputFile.close()    

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


stemmer = SnowballStemmer('english')
emotion_words_complete_dict = defaultdict(list)

for word in all_emo_words_7:
    #word = TextBlob(word)
    stemmed_word1 = stemmer.stem(word)
    for word2 in all_emo_words_7:
        #word2 = TextBlob(word2)
        stemmed_word2 = stemmer.stem(word2)
        if stemmed_word1 == stemmed_word2:
            emotion_words_complete_dict[stemmed_word1].append(word2)

len(emotion_words_complete_dict.keys())


#make ea list of values a set:

for key in emotion_words_complete_dict:
    emotion_words_complete_dict[key] = list(set(emotion_words_complete_dict[key]))


#test_dict = {}
for i, key in enumerate(emotion_words_complete_dict.keys()):
    if i < 10:        
        print key, emotion_words_complete_dict[key]
        #test_dict[key] = emotion_words_complete_dict[key]


#test_dict = {'uncomfort': ['uncomfortable'], 'consider': ['consideration', 
#'considerate', 'considerations', 'considerately'], 'dispirit': ['dispirits', 
#'dispiriting', 'dispirited', 'dispirit']}


#replace key with first or shortest word in values

for key in emotion_words_complete_dict.keys():
    new_key = emotion_words_complete_dict[key][0]
    for word in emotion_words_complete_dict[key]:
        if word[-2:] == 'ed':
            new_key = word  
    emotion_words_complete_dict[new_key] = emotion_words_complete_dict.pop(key)


#pickle this list so i don't have to go through above steps to get the emo_dict

#to pickle:
with open('emo_words_dict.pkl', 'w') as picklefile:
    pickle.dump(emotion_words_complete_dict, picklefile)
    
#to retrieve:
with open('emo_words_dict.pkl', 'r') as picklefile:
    emotion_words_complete_dict_2 = pickle.load(picklefile)

len(emotion_words_complete_dict_2)

#write complete emo words to file so can view
outputFile = open('all_emo_words_5.txt', 'w')  #creates a file object called outputFile. It also 
for key in emotion_words_complete_dict_2.keys():
    for word in emotion_words_complete_dict_2[key]: 
        outputFile.write(word+'\n')
outputFile.close()    



'emotion_words_storm_2.txt'
'emotion_words_clore_2.txt'

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


storm_words[:10]
clore_words[:10]
fletcher_words[:10]

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


clore_and_storm = sorted(list(set(storm_words + clore_words)))
len(clore_and_storm)
clore_and_storm[100:200]

clore_storm_vs_fletcher = show_words_not_in_list2(fletcher_words, clore_and_storm)
len(clore_storm_vs_fletcher)
clore_storm_vs_fletcher[900:]


#need to add guilty and guiltily, hate, hated, hates, hating
#'sad', sadden', 'saddened','saddening', 'saddens','sadistic', 'sadly', 'scary',
# 'weep', 'weeping', 'weeps', 'weepy', 'wept',
 













##############################################################################

# create variations on root words. lemmas are the variations.
# and also get dictionary to put synonums in same category

#if use this then i need to manually write all versions and then change back to root with this
lmtzr = WordNetLemmatizer()
lmtzr.lemmatize('worriedly', 'r')  #if do a verb, closer to what i want


#this gives words that are related, but more loose than synonyms
#e.g., worried will return fear. so SKIP this.
j = wn.synsets('worriedly')[0]
j.hyponyms()


#this is much closer to synonyms. might work for me???
for i in wn.synsets('worriedly'):
    if i.pos() in ['a', 's', 'v', 'n', 'r']: # If synset is adj or satelite-adj.
        for j in i.lemmas(): # Iterating through lemmas for each synset.
           print j
            

#not bad: takes the longer term and finds the root. e.g., worry. and synon. hmmm.
for word in wn.synsets('worriedly'):
    print word.lemma_names()
#anger = wn.lemmas('anger')


def get_synonyms(word, pos):
        synonyms = []
        for s in wn.synsets(word, pos):
                synonyms += s.lemma_names()
        return list(set(synonyms))

#this does work ok? though giving words that aren't really synonyms?
#I think i just want the lemmas. i.e., just group words from the same root.
#but would need to manually put all variations
get_synonyms('worriedly', 'n')  #kind of sucks. e.g, with adored it returns
#alls sorts of synonyms. but with adore it doestn't give any.


#synsets = wn.synsets('worried', pos='n')[0]
#l = synsets.lemmas()[0]
#x = l.derivationally_related_forms()

#SKIP
acknowledgment_synset = wn.synset('worried.a.01')
acknowledgment_lemma = acknowledgment_synset.lemmas()
for a in acknowledgment_lemma:
    print a.derivationally_related_forms()
    

WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'
 
 
 #think the synonym thing might not work that well for me
#better to just stick to grouping words with the same lemma/root
#might have to do that manually? though may be able to use:
lmtzr = WordNetLemmatizer()
lmtzr.lemmatize('worried', 'v')  #if do a verb, closer to what i want
#i.e., make list w all variations. and then use this above to create
#dictionaries to lumping together words from same lexeme?
#this this is only one that makes sense for now. create all the variations 
#i can and then plug the into this for all parts of speech to get lemma?
#and choose the shortest option? e.g., might get a few lemmas for diff parts
#of speech. just systematically choose the shortest.
#BUT THIS ISN'T WORKING THAT WELL EITHER! E.G., WON'T TAKE WORRIEDLY AND 
#MAKE IS LESS. SO MAYBE USE THIS TO SIMPLITFY A BIT??? OR JUST DO THE WHOLE
#THING BY HAND.



#more from the pattern module:
conjugate('irritated') # he / she / it. does this give me the root i want?!
f1 = wordnet.synsets('nervousness')[0]
f2 = wordnet.synsets('anxiety')[0]
f.synonyms
a.synonyms
wordnet.similarity(f1, f2) 

