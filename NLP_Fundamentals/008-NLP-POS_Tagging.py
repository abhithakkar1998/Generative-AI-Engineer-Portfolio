# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:06:36 2025

@author: kthakkara
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 11:49:07 2025

@author: kthakkara
"""

import nltk
from nltk.corpus import stopwords
#nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

speech = """
I have three visions for India. My FIRST VISION is that of FREEDOM. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us. MY SECOND VISION for India is DEVELOPMENT. For fifty years we have been a developing nation. It is time we see ourselves as a developed nation. We are among top five nations in the world in terms of GDP. I have a THIRD VISION. India must stand up to the world. Because I believe that unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand.
"""

documents = nltk.sent_tokenize(speech)

# Find POS Tags

for idx in range(len(documents)):
    words = nltk.word_tokenize(documents[idx])
    words = [word for word in words if word not in set(stopwords.words('english'))]
    pos_tag = nltk.pos_tag(words)
    print(f"For Sentence: {documents[idx]}\n POS tags: {pos_tag}")
    print("===============\n")

