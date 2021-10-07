---
layout: splash
title: "Edge Detection"
---

<br>

# Edge Detection

## Import Libraries

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

## Load Images

```python
img = cv2.imread('/content/target.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

## Patching Image

```python
new_img = np.zeros([img.shape[0]+2, img.shape[1]+2])
new_img[1:1+img.shape[0], 1:1+img.shape[1]] = img_gray
```

## Sobel Operator

```python
sobel_v = np.array([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]])

sobel_h = np.array([[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]])
```

## Convolutional Operation

```python
def Convolution(img, filter) :
  for r in range(img.shape[0]) :
    for c in range(img.shape[1]) :
      part = new_img[r:r+3, c:c+3]
      res[r, c] = np.sum(part * filter)
  return res
 ```
 
## Resulting Image
 
```python
res = np.zeros([img.shape[0], img.shape[1]])
res_v = Convolution(img, sobel_v)
res_h = Convolution(img, sobel_h)
res = np.sqrt(res_v * res_h)
```

## Display Image

```python
plt.figure(figsize = (10, 10))
plt.xticks([])
plt.yticks([])
plt.imshow(res)
```
