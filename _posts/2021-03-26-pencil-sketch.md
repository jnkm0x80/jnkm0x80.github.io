---
layout: splash
title: "Converting Image into Pencil Sketch"
---

<br>

<center><h1>Converting Image into Pencil Sketch</h1></center>

## Libraries Used

|    Library     | URL                        |
|:--------------:|:-------------------------- |
|   **OpenCV**   | https://opencv.org         |
|   **NumPy**    | https://numpy.org          |
| **Matplotlib** | https://matplotlib.org     |
| **Tensorflow** | https://www.tensorflow.org |

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_file
```

## Sample Image

For a sample image, I used this image in following [website](https://www.theatlantic.com/science/archive/2019/08/cookies-in-space/595396/).

```python
target = 'https://cdn.theatlantic.com/thumbor/jQ9I2Chl0yIOKEULfYCZMbONFuU=/0x221:5200x3146/960x540/media/img/mt/2019/08/Astronaut_Holding_DoubleTree_Cookie/original.jpg'
img = get_file('img', target)
```

We will convert this BGR format image into RGB and Grayscale format.

```python
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

Now we have following images :

```python
fig = plt.figure(figsize = (20, 20))

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(img_gray, cmap = 'gray')
```

![download](https://user-images.githubusercontent.com/91777895/138796658-e8b24406-33a1-493c-8be8-134f01a10088.png)

## Edge Detection

We will detect the edges of our image using Sobel operators. There are two matrices, one for horizontal edges and one for vertical edges.

```python
sobel_h = np.array([[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]])

sobel_v = np.array([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]])
```

We must use convolution for every pixel, making the resulting image of both matrices.

```python
def conv(img, filter) :

  img_r, img_c = img.shape
  res = np.zeros([img_r, img_c])

  patched_img = np.zeros([img_r+2, img_c+2])
  patched_img[1:1+img_r, 1:1+img_c] = img

  for r in range(img_r - 2) :
    for c in range(img_c - 2) :
        part = img[r:r+3, c:c+3]
        res[r, c] = np.sum(part * filter)

  return res
```

With the resulting images, we will find the final image with calculating Euclidean norm for them.

```python
img_h = conv(img_gray, sobel_h)
img_v = conv(img_gray, sobel_v)

img_res = np.sqrt(img_h ** 2 + img_v ** 2)
```

Followings are our horizontal, vertical, and final images :

```python
fig = plt.figure(figsize = (20, 20))

plt.subplot(132)
plt.imshow(img_h, cmap = 'gray')

plt.subplot(131)
plt.imshow(img_v, cmap = 'gray')

plt.subplot(133)
plt.imshow(img_res, cmap = 'gray')

plt.show()
```

![download (1)](https://user-images.githubusercontent.com/91777895/138796763-edb737a1-f762-47f6-a4c9-fe5edb99590b.png)

## Pencil Sketch Conversion

Lastly, we will convert the final image into the pencil sketch style based on intensity of white color.

```python
img_r, img_c = img_res.shape
img_pencil = np.zeros([img_r, img_c])
for r in range(img_r) :
  for c in range(img_c) :
    if img_res[r,c] > 150 :
      img_pencil[r,c] = 0
    else :
      img_pencil[r,c] = 255 - img_res[r,c]
```

Result :

```python
plt.figure(figsize = (15, 15))
plt.imshow(img_pencil, cmap = 'gray')
```

![download (2)](https://user-images.githubusercontent.com/91777895/138796831-596a6c2f-5528-440f-b237-c348f0d5464a.png)

