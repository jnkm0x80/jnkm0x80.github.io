---
layout: single
title: "Image Pixelation"
---

Code to Pixalate Images

## Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

```python
path = '/content/pixel.jpg'
img = cv2.imread(path)
```

```python
block = 10
imgH, imgW = (img.shape[0] // block, img.shape[1] // block)
resize = lambda x: cv2.resize(x, dsize = (imgW * block, imgH * block))
img = resize(img)
```


```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgR, imgG, imgB = cv2.split(img)
```


```python
def pixelate(M, block) :
  res = np.ones([M.shape[0], M.shape[1]])

  h_num = int(M.shape[0] / block)
  w_num = int(M.shape[1] / block)

  for i in range(h_num) :
    for j in range(w_num) :
      part = M[i*block:(i+1)*block, j*block:(j+1)*block]
      res[i*block:(i+1)*block, j*block:(j+1)*block] *= int(np.mean(part))

  return res
```


```python
new_imgs = [pixelate(M, block) for M in [imgR, imgG, imgB]]
result = cv2.merge((new_imgs[0], new_imgs[1], new_imgs[2]))
```


```python
plt.figure(figsize = (10, 10))
plt.xticks([])
plt.yticks([])
plt.imshow(result.astype('uint8'))
```
