---
layout: splash
title: "Blurring Faces with Deep Learning"
---

<br>

# Blurring Faces with Deep Learning

## Libraries Used


```python
import cv2, copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import AveragePooling2D
```

## Load Sample Image


```python
target = cv2.imread('/content/target.jpg')
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
```


```python
fig, axs = plt.subplots(1, 2)

axs[0].imshow(target)
axs[0].set_xticks([]); axs[0].set_yticks([])

axs[1].imshow(target_gray, cmap = 'gray')
axs[1].set_xticks([]); axs[1].set_yticks([])

plt.show()
```


![png](pixel_files/pixel_5_0.png)


## Detecting Faces


```python
model = cv2.CascadeClassifier(
  cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```


```python
faces = model.detectMultiScale(target_gray, 1.1, 4)
```


```python
target_copy = copy.deepcopy(target)
for x, y, w, h in faces :
  cv2.rectangle(target_copy, (x, y), (x+w, y+h), (255, 0, 0), 3)
plt.imshow(target_copy)
plt.xticks([]); plt.yticks([])
plt.show()
```


![png](pixel_files/pixel_9_0.png)


## Blurring Faces


```python
face_set = [target[y:y+h, x:x+w].astype('float32') for x, y, w, h in faces]
```


```python
plt.xticks([]); plt.yticks([])
plt.imshow(face_set[0].astype('int32'))
```




    <matplotlib.image.AxesImage at 0x7f66d3c76490>




![png](pixel_files/pixel_12_1.png)



```python
pooling = AveragePooling2D(pool_size = (20, 20), strides = (1, 1), padding = 'same')
pixel_face = []
```


```python
for face in face_set :
  r, c, layer = face.shape
  face = face.reshape(1, r, c, layer)
  pixel_face.append(pooling(face).numpy()[0].astype('int32'))
```


```python
for n in range(len(pixel_face)) :
  x, y, w, h = faces[n]
  target[y:y+h, x:x+w, :] = pixel_face[n]
```


```python
plt.figure()
plt.xticks([]); plt.yticks([])
plt.imshow(target)
```




    <matplotlib.image.AxesImage at 0x7f66d3c3e350>




![png](pixel_files/pixel_16_1.png)

