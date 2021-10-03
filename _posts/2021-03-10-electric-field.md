---
layout: splash
title: "Electric Field"
---

<br>

# Electric Field

## Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
```

## Initial Configuration

```python
xlim, ylim = ([-10, 10], [-10, 10])
density = 2
linewidth = 1
```

## Base Setup

```python
charges = [(-1e-9, (5, -3)), (3e-10, (-4, 6))]
X = np.linspace(xlim[0], xlim[1], 1000)
Y = np.linspace(ylim[0], ylim[1], 1000)
X, Y = np.meshgrid(X, Y)
```

## Electric Fields Strength

```python
def E_Field(P0, P, Q) :
  k = 8.99e+9
  r = np.sqrt((P[0] - P0[0]) ** 2 + (P[1] - P[0]) ** 2)
  return k * Q / (r ** 2)
```

## Calculate Total Electric Field Strength

```python
EX, EY = (0, 0)
for Q, (X0, Y0) in charges :
  E = E_Field([X0, Y0], [X, Y], Q)
  angle = np.arctan2(Y - Y0,  - X0)
  EX += E * np.cos(angle)
  EY += E * np.sin(angle)
```

## Visualizing Electric Field

```python
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111)
color = 2 * np.log(np.hypot(EX, EY))
ax.streamplot(X, Y, EX, EY, density = density, color = color,
              linewidth = linewidth, arrowstyle = '->', arrowsize = 1.5)
sign_color = {True: '#AA0000', False: '#0000AA'}
for Q, P0 in charges :
  ax.add_artist(Circle(P0, 0.25, color = sign_color[Q > 0]))
plt.show()
```
