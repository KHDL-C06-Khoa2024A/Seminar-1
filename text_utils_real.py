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
    
    sign = ["==>", "=>", "->", "\.\.\.", ">>"]
    digits = "\d+([\.,_]\d+)?"
    email = "[\w\.-]+@[\w\.-]+"
    web = urlmarker.WEB_URL_REGEX
    datetime = [
        "\d{1,2}\/\d{1,2}(\/\d+)?",
        "\d{1,2}-\d{1,2}(-\d+)?",
    ]
    word = "\w+"
    non_word = "[^\w\s]"
    abbreviations = [
        "[A-ZĐ]+\.",
        "Tp\.?",
        "Mr\.", "Mrs\.", "Ms\.",
        "Dr\.?", "ThS\.?", "TS\.?", "GS\.?", "PSG\.?"
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
def is_word(string):
  sign = ["==>", "=>", "->", "\.\.\.", ">>"]
  digits = "\d+([\.,_]\d+)?"
  email = "[\w\.-]+@[\w\.-]+"
  web = urlmarker.WEB_URL_REGEX
  datetime = [
      "\d{1,2}\/\d{1,2}(\/\d+)?",
      "\d{1,2}-\d{1,2}(-\d+)?",
  ]
  non_word = "[^\w\s]"
  abbreviations = [
      "[A-ZĐ]+\.",
      "Tp\.?",
      "Mr\.", "Mrs\.", "Ms\.",
      "Dr\.?", "ThS\.?", "TS\.?", "GS\.?", "PSG\.?"
  ]

  
  patterns = []
  patterns.extend(abbreviations)
  patterns.extend(sign)    
  patterns.extend(datetime)
  patterns.extend([web, email])
  patterns.extend([digits, non_word])
  patterns = "(" + "|".join(patterns) + ")"
  patterns = re.compile(patterns)
  return not bool(patterns.match(string))

 