# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 11:49:07 2025

@author: kthakkara
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
print(stopwords.fileids())
'''
available languages:
    ['albanian', 'arabic', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'catalan', 'chinese', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek', 'hebrew', 'hinglish', 'hungarian', 'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene', 'spanish', 'swedish', 'tajik', 'tamil', 'turkish']
'''

stopwords.words('english')

speech = """
I have three visions for India. In 3000 years of our history people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture and their history and tried to enforce our way of life on them. Why? Because we respect the freedom of others. That is why my FIRST VISION is that of FREEDOM. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.

We have 10 percent growth rate in most areas. Our poverty levels are falling. Our achievements are being globally recognised today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect? MY SECOND VISION for India is DEVELOPMENT. For fifty years we have been a developing nation. It is time we see ourselves as a developed nation. We are among top five nations in the world in terms of GDP.

I have a THIRD VISION. India must stand up to the world. Because I believe that unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand. My good fortune was to have worked with three great minds. Dr.Vikram Sarabhai, of the Dept. of Space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material. I was lucky to have worked with all three of them closely and consider this the great opportunity of my life.
"""

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

## Divide paragrahs into documents
documents = nltk.sent_tokenize(speech)
#print(documents)

## Apply stopwords and filter and then apply stemming
for idx in range(len(documents)):
    words = nltk.word_tokenize(documents[idx])
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in set(stopwords.words('english'))]
    documents[idx] = ' '.join(words)
    
speech_cleaned = ' '.join(documents)
print(speech_cleaned)


