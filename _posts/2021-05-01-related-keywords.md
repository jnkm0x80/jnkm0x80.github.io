---
layout: splash
title: "Related Keywords Using Graph Network"
---

<br>

<center><h1>Related Keywords Using Graph Network</h1></center>

## Libraries Used

|    Library     | URL                        |
|:--------------:|:-------------------------- |
|**Gensim*|https://radimrehurek.com/gensim|
|**NetworkX** |https://networkx.org |
|   **KeyBERT**   | https://maartengr.github.io/KeyBERT        |
| **Pandas** | https://pandas.pydata.org|

## Kaggle Dataset

```bash
!pip install kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download hsankesara/medium-articles
!unzip medium-articles.zip
```

## Get Articles

```python
import pandas as pd
train_docs = pd.read_csv('/content/articles.csv')
train_docs = [docs for docs in train_docs['text']]
```

## Load KeyBERT Model

```bash
!pip install keybert
```

```python
from keybert import KeyBERT
keybert_model = KeyBERT()
```

## Keyword Extraction

```python
def top_n_keywords(doc, ngram = 3, num = 15) :

  keyword, keyword_temp = [], []

  for n in range(ngram) :
    top_n_temp = 100
    while keyword_temp == [] :
      keyword_temp = keybert_model.extract_keywords(doc, keyphrase_ngram_range = (n+1, n+1), stop_words = 'english',
                                            use_mmr = True, diversity = 0.7, top_n = top_n_temp)
      top_n_temp = int(top_n_temp / 2)
    keyword += [(word[0], word[1] / pow(2, n)) for word in keyword_temp]
  keyword = sorted(keyword, key = lambda word: word[1], reverse = True)

  for n_word in [w[0] for w in keyword[:num] if len(w[0].split()) > 1] :
    if False not in [(word in [w[0] for w in keyword[:num]]) for word in n_word.split()] :
      keyword = [word for word in keyword if word[0] not in n_word.split()]

  return keyword[:num]
  ```
  
  ## Generating Data for Train Graph
  
  ```
  train_graph = {}
  for idx in range(len(train_docs)) :
    if idx % 50 == 0 :
      print('article {} processed'.format(idx))
    keywords = [word[0] for word in top_n_keywords(train_docs[idx])]

  for wkey in keywords :
    for wval in keywords :
      if wkey != wval :
        if wkey not in train_graph.keys() :
          train_graph[wkey] = {}
        if wval not in train_graph[wkey].keys() :
          train_graph[wkey][wval] = 1
        else :
          train_graph[wkey][wval] += 1
          
  max = 0
  for word in train_graph.keys() :
    for val in train_graph[word].values() :
      if val > max :
        max = val
        
  wedges = []
  for word in train_graph.keys() :
    for key, val in train_graph[word].items() :
      if (key, word, val) not in wedges :
        wedges.append((word, key, (max-val) * 100 + 1))
 ```

## Create Graph with NetworkX

```bash
!pip  install networkx
```

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_nodes_from(train_graph.keys())
G.add_weighted_edges_from(wedges)

for word in G.nodes() :
  G.nodes[word]['weight'] = len(train_graph[word].keys())
  G.nodes[word]['connection'] = [key for key in train_graph[word].keys()]

pos = nx.spring_layout(G, k = 3)
colors = [i/len(G.nodes) for i in range(len(G.nodes))]
nx.draw(G, pos, with_labels = False, node_size = 50, 
        node_color = colors, cmap = plt.cm.plasma, width = 0.25)

fig = plt.gcf()
fig.set_size_inches(18.5, 18.5)
plt.show()
 ```
 
 ![download](https://user-images.githubusercontent.com/91777895/142175569-dbde76d8-7b53-46f9-8379-4d4bbc2bd7b6.png)
 
## Applying DeepWalk to Graph

```
import random

def random_walk(graph, node, steps) :
  path = [str(node)]
  target = node
  for _ in range(steps) :
    neighbors = list(nx.all_neighbors(graph, target))
    target = random.choice(neighbors)
    path.append(str(target))
  return path
 random_walk_total = []
 
for node in G.nodes() :
  random_walk_total += [random_walk(G, node, 20) for _ in range(100)]
  
from gensim.models import Word2Vec

model = Word2Vec(random_walk_total, size = 32, window = 5, min_count = 1, sg = 1)
```
 
 ## Final Model and Recommendation
 
 ```python
def recommendation(word, topn = 30) :
  return [(n for n in model.wv.most_similar(word, topn = topn))]
  
word = random.choice(list(G.nodes()))
print(word)
list(recommendation(word)[0])[:10]
```

Output :

```python
calculating
[('preprocessing', 0.993146538734436),
 ('crm', 0.9924346208572388),
 ('inconsistencies', 0.9908812046051025),
 ('value', 0.9908706545829773),
 ('attributes', 0.9878985285758972),
 ('incomplete', 0.9843828678131104),
 ('ignored', 0.9843740463256836),
 ('students', 0.8592240214347839),
 ('predicting', 0.8086998462677002),
 ('useful', 0.7909170389175415)]
```
