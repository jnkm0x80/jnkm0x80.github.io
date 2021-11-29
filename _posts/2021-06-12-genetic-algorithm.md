---
layout: splash
title: "Drawing Image Using Genetic Algorithm"
---

<br>

<center><h1>Drawing Image Using Genetic Algorithm</h1></center>

## Important Libraries Used

| Library | URL |
|:-------:|:----|
| **OpenCV** | https://opencv.org |
| **NumPy** |https://numpy.org |
| **scikit-image**|https://scikit-image.org|

```python
import cv2, os, sys
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import random as rd
from copy import deepcopy
from skimage.metrics import mean_squared_error
```

## Sample Image

```python
img = cv2.imread('image.jpg')
h, w, c = img.shape
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

![download](https://user-images.githubusercontent.com/91777895/143839731-ac079de2-0189-48c7-bbaa-2a77fb46ed18.png)

## Parameters for Genetic Process

```python
n_genes, n_population = (50, 50)
prob_mut, prob_add, prob_rmv = (0.01, 0.3, 0.2)
min_rad, max_rad = (5, 15)
```

## Setup Circle as Gene in Drawing

```python
class Gene :
  
  def __init__(self) :
    self.radius = rd.randint(min_rad, max_rad)
    self.center = np.array([rd.randint(0, w), rd.randint(0, h)])
    self.colors = np.array([rd.randint(0, 255) for _ in range(3)])
  
  def mutate(self) :
    size = max(1, int(round(rd.gauss(15, 4)))) / 100
    x = rd.uniform(0, 3)
    if x < 1 :
      self.radius = np.clip(rd.randint(
          int(self.radius * (1 - size)),
          int(self.radius * (1 + size))
      ), 1, 100)
    elif x < 2 :
      self.center = np.array([
        np.clip(rd.randint(
            int(self.center[0] * (1 - size)),
            int(self.center[0] * (1 + size))
        ), 0, w),
        np.clip(rd.randint(
            int(self.center[1] * (1 - size)),
            int(self.center[1] * (1 + size))
        ), 0, h)
      ])
    else :
      self.colors = np.array([
         np.clip(rd.randint(
             int(self.colors[i] * (1 - size)),
             int(self.colors[i] * (1 + size))
         ), 0, 255) for i in range(3)
      ])
```

## Similarity Between Output and Base Image

```python
def compute_fitness(genome):
  out = np.ones((h, w, c), dtype = np.uint8) * 255
  for gene in genome:
    cv2.circle(out, center = tuple(gene.center), 
                    radius = gene.radius, 
                    color = (int(gene.colors[0]), 
                             int(gene.colors[1]), 
                             int(gene.colors[2])), thickness = -1)
  fitness = 255.0 / mean_squared_error(img, out)
  return fitness, out
```

## Occurrence of Events (Mutation, Addition, Removal)

```python
def compute_population(g):
  genome = deepcopy(g)

  if len(genome) < 200:
    for gene in genome:
      if rd.uniform(0, 1) < prob_mut:
        gene.mutate()
  else:
    for gene in rd.sample(genome, k=int(len(genome) * prob_mut)):
      gene.mutate()

  if rd.uniform(0, 1) < prob_add:
    genome.append(Gene())
  if len(genome) > 0 and rd.uniform(0, 1) < prob_rmv:
    genome.remove(rd.choice(genome))

  new_fitness, new_out = compute_fitness(genome)
  return new_fitness, genome, new_out
```

## Running Genetic Algorithm

```python
if __name__ == '__main__':
  os.makedirs('result', exist_ok=True)
  p = mp.Pool(mp.cpu_count() - 1)

  best_genome = [Gene() for _ in range(n_genes)]
  best_fitness, best_out = compute_fitness(best_genome)
  n_gen = 0

  while True:
    try:
      results = p.map(compute_population, 
                      [deepcopy(best_genome)] * n_population)
    except KeyboardInterrupt:
      p.close()
      break

    results.append([best_fitness, best_genome, best_out])
    new_fitnesses, new_genomes, new_outs = zip(*results)

    best_result = sorted(zip(
        new_fitnesses, new_genomes, new_outs
    ), key=lambda x: x[0], reverse=True)
    best_fitness, best_genome, best_out = best_result[0]

    print('Generation #%s, Fitness %s' % (n_gen, best_fitness))

    cv2.imwrite('result/res_%d.jpg' % (n_gen), best_out)

    n_gen += 1
```

## Output

![sample](https://user-images.githubusercontent.com/91777895/143840005-551bc73d-0a29-48be-b956-193a6aef7a65.gif)

Each Frame Recorded per 1,000 Iteration through 25,000 Loops
