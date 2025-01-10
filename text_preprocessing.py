import json, os, ast, re, sys
import unicodedata as ud

from text_utils import remove_stopwords, token_sylabling, is_word
from utils import load_file_with_newline
import urlmarker

# Load all dictionary 
dict_path = '/Users/scottnguyen/Desktop/Project_06/vfnd-fake-news-experiences/Dictionaries/'


bi_dict = load_file_with_newline(os.path.join(dict_path, 'bi_gram.txt'))
tri_dict = load_file_with_newline(os.path.join(dict_path, 'tri_gram.txt'))
four_dict = load_file_with_newline(os.path.join(dict_path, 'four_gram.txt'))

stopwords = load_file_with_newline(os.path.join(dict_path, 'Stopwords_vi.txt'))

bi_gram = [x.lower() for x in bi_dict]
tri_gram = [x.lower() for x in tri_dict]
four_gram = [x.lower() for x in four_dict]


# LongestMatching use for word segmentation
def LongestMatching(token, bi_gram, tri_gram, four_gram):
    #token_len: Length of syllable list
    token_len = len(token)
    cur_id = 0
    word_list = []
    
    #done: True when cur_id reach 
    done = False
    
    while (cur_id < token_len) and (not done):
        cur_word = token[cur_id]
        if(cur_id >= token_len - 1):
            word_list.append(cur_word)
            done = True
        else:
            next_word = token[cur_id + 1]
            bi_word = " ".join([cur_word.lower(), next_word.lower()])
            if(cur_id >= token_len - 2):
                if bi_word in bi_gram:
                    word_list.append("_".join([cur_word, next_word]))
                    cur_id  = cur_id + 2
                else: 
                    word_list.append(cur_word)
                    cur_id  = cur_id + 1
                        
            else: 
                bi_next_word = token[cur_id + 2]
                tri_word = " ".join([bi_word, bi_next_word.lower()])
                if(cur_id >= token_len - 3):
                    if tri_word in tri_gram:
                        word_list.append("_".join([cur_word, next_word, bi_next_word]))
                        cur_id  = cur_id + 3
                    elif bi_word in bi_gram:
                        word_list.append("_".join([cur_word, next_word]))
                        cur_id = cur_id + 2
                    else:
                        word_list.append(cur_word)
                        cur_id = cur_id + 1
                else:
                    tri_next_word = token[cur_id + 3]
                    four_word = " ".join([tri_word, tri_next_word.lower()])
                    if four_word in four_gram:
                        word_list.append("_".join([cur_word, next_word, bi_next_word, tri_next_word]))
                        cur_id  = cur_id + 4
                    elif tri_word in tri_gram:
                        word_list.append("_".join([cur_word, next_word, bi_next_word]))
                        cur_id  = cur_id + 3
                    elif bi_word in bi_gram:
                        word_list.append("_".join([cur_word, next_word]))
                        cur_id = cur_id + 2
                    else:
                        word_list.append(cur_word)
                        cur_id = cur_id + 1
    return word_list


def text_preprocessing(text):
    token_list = token_sylabling(text)
    word_list = LongestMatching(token_list, bi_gram, tri_gram, four_gram)
    remove_stopword_list = remove_stopwords(word_list, stopwords)
    remove_non_word = [s for s in remove_stopword_list if is_word(s)]
    new_text = ' '.join(remove_non_word)
    return new_text.lower()