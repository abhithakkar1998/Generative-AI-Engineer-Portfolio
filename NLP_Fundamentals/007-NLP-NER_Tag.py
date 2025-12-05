# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:22:16 2025

@author: kthakkara
"""

import nltk
#nltk.download("maxent_ne_chunker")
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

sentence = "It is time we see ourselves as a developed nation. We are among top five nations in the world in terms of GDP."

words = nltk.word_tokenize(sentence)

tag_elements = nltk.pos_tag(words)

nltk.ne_chunk(tag_elements).draw()