# This file stores a variety of helper functions used in functions that produce
# feature vectors when called. The purpose of the varying feature functions is
# to test the performance of a model using Ablation Analysis, which is the 
# process of testing how well a model is impacted by the removal of a single
# feature.

################################################################################
################################## Imports #####################################
################################################################################
import nltk
from datetime import datetime
from nltk.tokenize import word_tokenize

################################################################################
############################## Helper Methods ##################################
################################################################################

# Returns the MSE of a list of preditions & labels
def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

# Returns a list the frequencies of a given title's Parts of Speech
def parts_of_speech(title):
    # Tokenize the words in the title
    tokens = word_tokenize(title)
    
    # Turns each token into a pair with its value and Part of Speech label
    # More Info Here: https://realpython.com/nltk-nlp-python/#tagging-parts-of-speech
    pos = nltk.pos_tag(tokens)
    
    # Map the generalized Parts of Speech to their frequency in the title
    frequencies = {
        "Adjectives":0,
        "Nouns":0,
        "Adverbs":0,
        "Pronouns":0,
        "Verbs":0,
        "Determiners":0
    }
    
    # Count the frequencies of each Part of Speech generalizing to 7 categories
    for pair in pos:
        if pair[1].startswith("JJ"):
            frequencies["Adjectives"] += 1
        elif pair[1].startswith("NN"):
            frequencies["Nouns"] += 1
        elif pair[1].startswith("RB"):
            frequencies["Adverbs"] += 1
        elif pair[1].startswith("PRP"):
            frequencies["Pronouns"] += 1
        elif pair[1].startswith("VB"):
            frequencies["Verbs"] += 1
        elif pair[1].startswith("DT"):
            frequencies["Determiners"] += 1
        else:
            pass

    return list(frequencies.values())

# Returns a one-hot encoding (OHE) of the hour of day, and weekday
# OHE allows for encoding a n-length list of binary features in n - 1 space
def one_hot_encoding_time(unixtime):
    hour = [0] * 23
    week = [0] * 6
    
    # Get the local time of the given unix timestamp
    time = datetime.fromtimestamp(int(float(unixtime)))
    
    # One hot encode the hour (hour 0 is just a list of 0's)
    # https://docs.python.org/3/library/datetime.html#datetime.datetime.hour
    if time.hour != 0:
        hour[time.hour - 1] = 1
    
    # One hot encode the weekday (day 0 is just a list of 0's)
    # https://docs.python.org/3/library/datetime.html#datetime.date.weekday
    if time.weekday() != 0:
        week[time.weekday() - 1] = 1
        
    return hour + week
    
# Helper function returns a list that represents the presence of popular words
def popular_words(title, n, n_popular_words):
    words = [0] * n
    
    for word in word_tokenize(title):
        if word in n_popular_words:
            words[n_popular_words.index(word)] = 1
    
    return words

################################################################################
############################# Feature Methods ##################################
################################################################################


# Creates a feature vector for a given row of data
def feature_all(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_score(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    # feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_num_com(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    # feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_len_char(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    # feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_len_word(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    # feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_oc(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    # feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_pos(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    # feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_ohe(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    # feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat

# Creates a feature vector for a given row of data
def feature_exc_popular_word(datum, n_popular_words, n):
    feat = [1]
    
    # Add a feature for the score (price of awards given)
    feat.append(int(datum['score']))
    
    # Add a feature for the number of comments
    feat.append(int(datum['number_of_comments']))
    
    # Add a feature for character length of title
    feat.append(len(datum['title']))
    
    # Add a feature for word length of title
    feat.append(len(word_tokenize(datum['title'])))
    
    # Add a binary feature for if the content is declared original (OC)
    feat.append(1) if "[oc]" in datum['title'].lower() else feat.append(0)
    
    # Add features for the frequencies of generalized Parts of Speech
    feat.extend(parts_of_speech(datum['title']))
    
    # Add features for the one-hot encoding of the Hour and Weekday
    feat.extend(one_hot_encoding_time(datum['unixtime']))
    
    # Add feature list for the presence of any of the n-most popular words
    # feat.extend(popular_words(datum['title'], n, n_popular_words))
    
    return feat
