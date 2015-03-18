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
    
#to retrieve pickle:
with open('clore_and_storm_words_Mar19_dict.pkl', 'r') as picklefile:
    clore_and_storm_Mar19_dict = pickle.load(picklefile)


#now that have dict of emo words...




#turn these into functions:

#in each set of text docs
#change all words in waking reports to lowercase
def corpus_lowercase(corpus):
    corpus_lower =[]
    for report in corpus:
        textblob_report = TextBlob(report)
        new_report = ' '.join([word.lower() for word in textblob_report.words]) 
        corpus_lower.append(new_report) 
    return corpus_lower


#corret spelling in waking reports
def corpus_spelling_correct(corpus):
    corpus_spell_correct =[]
    for report in corpus_spell_correct:
        textblob_report = TextBlob(report)
        report_spelled = textblob_report.correct()
        corpus_spell_correct.append(report_spelled)
    return corpus_spell_correct
    
    
#replace all emotion words in waking report corpus with the root word. 
#i.e, replace scare and scary with scared. 
def replace_emo_words_w_root(corpus, emotion_to_root_dict):
    corpus_replaced_emotions = []
    for report in corpus:   
        for key in emotion_to_root_dict.keys():
            for word in emotion_to_root_dict[key]:
                report = report.replace(word, key)
        corpus_replaced_emotions.append(report)
    return corpus_replaced_emotions


#create dict where emotion_complete category is the key and the values are whether absent or present in each report (waking)
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
def get_emotion_ratios(alphabetical_emotion_counts_corpus1, alphabetical_emotion_counts_corpus2):
    """Takes list of emotions and their counts sorted alphabetically and computes emotion ratios.
    Then sorts these emotion ratios from highest to lowest"""
    emotions_ratio_list = [] 
    for i in range(len(alphabetical_emotion_counts_corpus1)):
        emotion = alphabetical_emotion_counts_corpus1[i][0]
        ratio = float((alphabetical_emotion_counts_corpus1[i][1] + 10)) / float((alphabetical_emotion_counts_corpus2[i][1] + 10))
        emotions_ratio_list.append([emotion, ratio])
    sorted_emotion_ratios = sorted(emotions_ratio_list, key=get_key, reverse=True)
    return sorted_emotion_ratios


#compute ratio of emotions in dream reports over dream reports
waking_emotion_complete_counts_sorted_alphabetically = sort_emotion_counts_alphabetically(waking_emotions_complete_dictionary)
dream_emotion_complete_counts_sorted_alphabetically = sort_emotion_counts_alphabetically(dream_emotions_complete_dictionary)

#compute dream words to real-life words ratio
dream_to_wake_emotion_complete_ratios = get_emotion_ratios(dream_emotion_complete_counts_sorted_alphabetically, waking_emotion_complete_counts_sorted_alphabetically)

#plot
X = [word[0] for word in dream_to_wake_emotion_complete_ratios[:25]]
Y = [freq[1] for freq in dream_to_wake_emotion_complete_ratios[:25]]

fig = plt.figure(figsize=(15, 5))  #add this to set resolution: , dpi=100
sns.barplot(x = np.array(range(len(X))), y = np.array(Y))
sns.despine(left=True)
plt.title('Emotion-words Most Representative of Dreams', fontsize=17)
plt.xticks(rotation=75)
plt.xticks(np.array(range(len(X))), np.array(X), rotation=75, fontsize=15)
plt.ylim(1, 3.05)
plt.ylabel("Frequency in dreams relative to real events", fontsize=15)




#maybe to add? 
'unworthy', 
'worthy', 
 u'unsettle',
 u'unsettled',
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
 'remorsefully']
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
 'judgmentally']
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
 'bleakness',

















