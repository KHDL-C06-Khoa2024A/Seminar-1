import unicodedata as ud
import sys
import re
import urlmarker

"""
  token_sylabling: String -> Sylable Token 
  input: text - Unicode String
  output: List() of token
"""
def token_sylabling(text):    
    text = ud.normalize("NFC", text)
    
    sign = [r"==>", r"=>", r"->", r"\.\.\.", r">>"]
    digits = r"\d+([\.,_]\d+)?"
    email = r"[\w\.-]+@[\w\.-]+"
    web = urlmarker.WEB_URL_REGEX
    datetime = [
        r"\d{1,2}\/\d{1,2}(\/\d+)?",
        r"\d{1,2}-\d{1,2}(-\d+)?",
    ]
    word = r"\w+"
    non_word = r"[^\w\s]"
    abbreviations = [
        r"[A-ZĐ]+\.",
        r"Tp\.?",
        r"Mr\.", r"Mrs\.", r"Ms\.",
        r"Dr\.?", r"ThS\.?", r"TS\.?", r"GS\.?", r"PSG\.?"
    ]
    
    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(sign)    
    patterns.extend(datetime)
    patterns.extend([web, email])
    patterns.extend([digits, non_word, word])
    patterns = "(" + "|".join(patterns) + ")"
    
    if sys.version_info < (3, 0):
        patterns = patterns.decode("utf-8")
    tokens = re.findall(patterns, text, re.UNICODE)
    return [token[0] for token in tokens]

"""
  remove_stopwords: remove "stopwords" from "paragraph"
  input: 
    + stopwords: Set() of stopwords
    + paragraph: List() of word in paragraph
  output: List() of words after remove stopwords
"""
def remove_stopwords(paragraph, stopwords):
    new_para = []
    for word in paragraph:
        if not word in stopwords:
            new_para.append(word)
    return new_para

"""
  remove_punc: remove punctuation from text
  input:
    + text: String Type
  output: text after remove all punctuations
"""
def remove_punc(text):
    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                        if ud.category(chr(i)).startswith("P"))
    return text.translate(tbl)

"""
  is_word: Check if <string> is a word
"""
# def is_word(string):
#     sign = [r"==>", r"=>", r"->", r"\.\.\.", r">>"]
#     digits = r"\d+([\.,_]\d+)?"
#     email = r"[\w\.-]+@[\w\.-]+"
#     web = urlmarker.WEB_URL_REGEX
#     datetime = [
#         r"\d{1,2}\/\d{1,2}(\/\d+)?",
#         r"\d{1,2}-\d{1,2}(-\d+)?",
#     ]
#     non_word = r"[^\w\s]"
#     abbreviations = [
#         r"[A-ZĐ]+\.",
#         r"Tp\.?",
#         r"Mr\.", r"Mrs\.", r"Ms\.",
#         r"Dr\.?", r"ThS\.?", r"TS\.?", r"GS\.?", r"PSG\.?"
#     ]
    
#     patterns = []
#     patterns.extend(abbreviations)
#     patterns.extend(sign)    
#     patterns.extend(datetime)
#     patterns.extend([web, email])
#     patterns.extend([digits, non_word])
#     patterns = "(" + "|".join(patterns) + ")"
#     patterns = re.compile(patterns)
#     return not bool(patterns.match(string))

import re
import urlmarker  # Assuming urlmarker.py is in the same directory or properly installed.

def is_word(string):
    # Patterns without the inline flags
    sign = [r"==>", r"=>", r"->", r"\.\.\.", r">>"]
    digits = r"\d+([\.,_]\d+)?"
    email = r"[\w\.-]+@[\w\.-]+"
    
    # Remove (?i) from inline regex and handle the case with the flags argument
    web = urlmarker.WEB_URL_REGEX.replace("(?i)", "")  # Remove inline case-insensitive flag
    datetime = [
        r"\d{1,2}\/\d{1,2}(\/\d+)?",
        r"\d{1,2}-\d{1,2}(-\d+)?",
    ]
    non_word = r"[^\w\s]"
    abbreviations = [
        r"[A-ZĐ]+\.",
        r"Tp\.?",
        r"Mr\.", r"Mrs\.", r"Ms\.",
        r"Dr\.?", r"ThS\.?", r"TS\.?", r"GS\.?", r"PSG\.?"
    ]

    # Combine all patterns into a single regex with alternation
    combined_patterns = []
    combined_patterns.extend(abbreviations)
    combined_patterns.extend(sign)
    combined_patterns.extend(datetime)
    combined_patterns.extend([web, email])
    combined_patterns.extend([digits, non_word])

    # Create a single regex pattern with all parts
    combined_pattern = "(" + "|".join(combined_patterns) + ")"

    # Compile the pattern correctly using flags parameter
    compiled_pattern = re.compile(combined_pattern, flags=re.IGNORECASE)  # Set case insensitivity with flags

    # Check if the input string matches any of the compiled patterns
    return not bool(compiled_pattern.match(string))

