---
layout: splash
title: "Keyword Extraction from Text"
---

<br>

<center><h1>Keyword Extraction from Text</h1></center>

## Libraries Used

|    Library     | URL                        |
|:--------------:|:-------------------------- |
|   **KeyBERT**   | https://maartengr.github.io/KeyBERT        |

```bash
!pip install keybert
```

## Load KeyBERT Model

```python
from keybert import KeyBERT

model = KeyBERT()
```

## Sample Text

For the sample text, I used an article [Neuron Bursts Can Mimic Famous AI Learning Strategy](https://www.quantamagazine.org/brain-bursts-can-mimic-famous-ai-learning-strategy-20211018/) from Quanta Magazine website.

```python
doc = '''Every time a human or machine learns how to get better at a task, a trail of evidence is left behind.
A sequence of physical changes — to cells in a brain or to numerical values in an algorithm — underlie the improved performance.
But how the system figures out exactly what changes to make is no small feat.
It’s called the credit assignment problem, in which a brain or artificial intelligence system must pinpoint which pieces in its pipeline are responsible for errors and then make the necessary changes. 
Put more simply: It’s a blame game to find who’s at fault.

...

“There’s got to be details that we don’t have, and we have to make the model better,” said Naud. 
“The main goal of the paper is to say that the sort of learning that machines are doing can be approximated by physiological processes.”

AI researchers are also excited, since figuring out how the brain approximates backpropagation could ultimately improve how AI systems learn, too. 
“If we understand it, then this may eventually lead to systems that can solve computational problems as efficiently as the brain does,” said Marcel van Gerven, chair of the artificial intelligence department at the Donders Institute at Radboud University in the Netherlands.

The new model suggests the partnership between neuroscience and AI could also move beyond our understanding of each one alone and instead find the general principles that are necessary for brains and machines to be able to learn anything at all.

“These are principles that, in the end, transcend the wetware,” said Larkum.'''
```

## Function for Keyword Extraction

```python
def top_n_keywords(doc, ngram = 5, num = 10) :

  keyword = []; keyword_temp = []

  for n in range(ngram) :

    top_n_temp = 100

    while keyword_temp == [] :

      keyword_temp = model.extract_keywords(doc, keyphrase_ngram_range = (n+1, n+1), stop_words = 'english',
                                            use_mmr = True, diversity = 0.7, top_n = top_n_temp)
      
      top_n_temp = int(top_n_temp / 2)
    
    keyword += [(word[0], word[1] / pow(2, n)) for word in keyword_temp]

  keyword = sorted(keyword, key = lambda word: word[1], reverse = True)

  for n_word in [w[0] for w in keyword[:num] if len(w[0].split()) > 1] :

    if False not in [(word in [w[0] for w in keyword[:num]]) for word in n_word.split()] :

      keyword = [word for word in keyword if word[0] not in n_word.split()]

  return keyword[:num]
 ```
 
 ## Result
 
 ```python
 top_n_keywords(doc, ngram = 3, num = 15)
 ```
 
 Output :
 
 ```python
 [('ai', 0.4855),
 ('backpropagation', 0.4215),
 ('neuroscientists', 0.3475),
 ('networks', 0.3441),
 ('inputs', 0.3012),
 ('credit', 0.2942),
 ('blame', 0.2765),
 ('efficiently', 0.2646),
 ('researchers', 0.2504),
 ('workhorse', 0.2436),
 ('ai', 0.24275),
 ('features', 0.2404),
 ('algorithm', 0.2376),
 ('processes', 0.216),
 ('ideas', 0.2138)]
```
