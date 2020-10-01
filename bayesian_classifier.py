# to write a bayesian classifier for spam e-mails
import numpy as np
import math

def load_dataSet():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return posting_list, class_vec 

def create_vocab_list (dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

def word_set2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else: print("the word: %s is not in my vocab_list!" % word)
    return return_vec

# have a test on these functions
list_posts, list_class = load_dataSet()
my_vocab_list = create_vocab_list(list_posts)
word_set2vec(my_vocab_list, list_posts[0])

np.zeros(10)

def train_NB(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words); p1_num = np.ones(num_words)
    p0_denom = 2.0; p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = math.log(p1_num / p1_denom)
    p0_vect = math.log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive

train_mat = []
for post_in_doc in list_posts:
    train_mat.append(word_set2vec(my_vocab_list, post_in_doc))

print(train_mat)
print(list_class)