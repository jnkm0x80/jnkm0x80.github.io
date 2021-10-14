---
layout: splash
title: "Face Blurring with Deep Learning"
---

<br>

<center><h1>Face Blurring with Deep Learning</h1></center>

## Introduction

Individual privacy has been a concerning issue in today's society, especially on social media platforms like YouTube or Facebook. One solution is to blur the faces of people in videos and images. It made me think of the way to classify human faces in an image and blur them automatically.

## Libraries Used

|    Library     | URL                        |
|:--------------:|:-------------------------- |
|   **OpenCV**   | https://opencv.org         |
|   **NumPy**    | https://numpy.org          |
| **Matplotlib** | https://matplotlib.org     |
| **Tensorflow** | https://www.tensorflow.org |

``` python
import cv2, copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
```

## Sample Image

For a sample image, I used the image of Real Madrid soccer team players in [this webpage](https://www.managingmadrid.com/2021/9/24/22690975/managing-madrid-podcast-how-real-is-this-real-madrid-plus-legacies-of-benzema-and-courtois). Now, let's load this with the OpenCV library.

```python
target = cv2.imread('/content/target.jpg')
```

However, since OpenCV loads images in BGR format, we must convert our loaded to RGB format. Also, let's create a grayscale of this image for the sample.

```python
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_gray = cv2.cvtColor(target, cv2.RGB2GRAY)
```

Let's check if we successfully retrieved our images.

```python
fig, axs = plt.subplots(1, 2, figsize = (20, 20))

axs[0].imshow(target)
axs[0].set_xticks([]); axs[0].set_yticks([])

axs[1].imshow(target_gray, cmap = 'gray')
axs[1].set_xticks([]); axs[1].set_yticks([])
 
plt.show()
```

Output :

![[download (4).png]]

## Face Detection

For our model, we will use Haar Cascades of OpenCV.

```python
model = cv2.CascadeClassifier(
 cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

Let's apply this model to our target image (grayscale).

```python
faces = model.detectMultiScale(target_gray, 1.1, 4)
```

What we'll get from this code is the set of x coordinate, y coordinate of an initial point, width, and height of face the model detected in the image. Here is the result.

```python
target_copy = copy.deepcopy(target)
for x, y, w, h in faces :
 cv2.rectangle(target_copy, (x, y), (x+w, y+h), (255, 0, 0), 3)

plt.imshow(target_copy)
plt.xticks([]); plt.yticks([])
plt.show()
```

Output :

![[download (5).png]]

## Blurring Faces

We will make the list of detected images.

```python
face_set = [target[y:y+h, x:x+w].astype('float32') for x, y, w, h in faces]
```

Each element in the list will look like this.

```python
plt.xticks([]); plt.yticks([])
plt.imshow(face_set[0].astype('int32'))
```

Output :

![[download (6).png]]

Now, we will load the **Average Pooling** model of Keras.

```python
pooling = AveragePooling2D(pool_size = (20, 20), strides = (1, 1), padding = 'same')
```

Apply the model to every face image in the list, appending them to another list pixel_face.

```python
pixel_face = []
for face in face_set :
 r, c, layer = face.shape
 face = face.reshape(1, r, c, layer)
 pixel_face.append(pooling(face).numpy()[0].astype('int32'))
```

Now, we will paste blurred face images over the original faces in our target image.

```python
for n in range(len(pixel_face)) :
 x, y, w, h = faces[n]
 target[y:y+h, x:x+w, :] = pixel_face[n]
```

As a result, we get following image.

```python
plt.figure()
plt.xticks([]); plt.yticks([])
plt.imshow(target)
```

Output :

![[download (7).png]]
