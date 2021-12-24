---
layout: splash
title: "Mask Detection Using Machine Learning"
---

<br>

<center><h1>Mask Detection Using Machine Learning</h1></center>

## Mount to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Libraries Used

| Library | URL |
|:-------:|:----|
| **OpenCV** | https://opencv.org |
|**cvlib**|https://www.cvlib.net|
| **NumPy** |https://numpy.org |
|**Matplotlib**|https://matplotlib.org|
| **TensorFlow**|https://www.tensorflow.org|

```python
!pip install cvlib

import os, cv2
import cvlib as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
```

## Load Image Dataset

[Face Mask Lite Dataset](https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset) from Kaggle

```python
path1 = './drive/MyDrive/without_mask/'
path2 = './drive/MyDrive/with_mask/'

flist1 = os.listdir(path1)
flist2 = os.listdir(path2)
fnum = len(flist1) + len(flist2)
```

## Preprocessing Image Dataset

```python
num = 0
images = np.float32(np.zeros((fnum, 224, 224, 3)))
labels = np.float64(np.zeros((fnum, 1)))

for imname in flist1 :

  impath = path1 + imname
  img = load_img(impath, target_size = (224, 224))

  x = img_to_array(img)
  x = np.expand_dims(x, axis = 0)
  x = preprocess_input(x)

  images[num, :, :, :] = x
  labels[num] = 0
  num += 1
  
for imname in flist2 :

  impath = path2 + imname
  img = load_img(impath, target_size = (224, 224))

  x = img_to_array(img)
  x = np.expand_dims(x, axis = 0)
  x = preprocess_input(x)

  images[num, :, :, :] = x
  labels[num] = 1
  num += 1
```

## Randomize Dataset for Training

```python
elem = labels.shape[0]
idx = np.random.choice(elem, size = elem, replace = False)

labels = labels[idx]
iamges = images[idx]

n_train = int(np.round(labels.shape[0] * 0.8))
n_test = int(np.round(labels.shape[0] * 0.2))

train_image = images[0:n_train, :, :, :]
test_image = images[n_train:, :, :, :]

train_label = labels[0:n_train]
test_label = labels[n_train:]
```

## Create Pre-Trained Model

```python
imshape = (224, 224, 3)

base = ResNet50(input_shape = imshape, weights = 'imagenet', include_top = False)
base.trainable = False
base.summary()

flatten_layer = Flatten()
dense_layer1 = Dense(128, activation = 'relu')
normal_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation = tf.nn.sigmoid)

model = Sequential([base, flatten_layer,
                    dense_layer1, normal_layer1,
                    dense_layer2])

base_lr = 0.001
model.compile(optimizer = tf.keras.optimizers.Adam(lr = base_lr),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()
model.fit(train_image, train_label, epochs = 10, batch_size = 16, 
          validation_data = (test_image, test_label))
model.save('model.h5')
```

## Loading Model to Detect Mask

```python
model = load_model('model.h5')
model.summary()

img1 = cv2.imread('with_mask.jpg')
img2 = cv2.imread('without_mask.jpg')

fig, axs = plt.subplots(figsize = (20, 20), nrows=1, ncols=2)
axs[0].imshow(img1[:, :, ::-1])
axs[1].imshow(img2[:, :, ::-1])
```

### Output :

![download](https://user-images.githubusercontent.com/91777895/147311784-f3907885-c6ec-4215-bfb6-917229397746.png)

```python
def detect_mask(img) :

  face, confidence = cv.detect_face(img)

  for idx, frame in enumerate(face) :
    
    (X0, Y0, X1, Y1) = tuple(frame)
    condition = [0 <= X0 <= img.shape[1],
                0 <= X1 <= img.shape[1],
                0 <= Y0 <= img.shape[0],
                0 <= Y1 <= img.shape[0]]
    if False not in condition :
      face = img[Y0:Y1, X0:X1]
      face_image = cv2.resize(face, (224, 224), 
                              interpolation = cv2.INTER_AREA)

      x = img_to_array(face_image)
      x = np.expand_dims(x, axis = 0)
      x = preprocess_input(x)

      pred = model.predict(x)

      if pred < 0.5 :
        cv2.rectangle(img, (X0, Y0), (X1, Y1), (0, 0, 255), 4)
      else :
        cv2.rectangle(img, (X0, Y0), (X1, Y1), (0, 255, 0), 4)

    return img
```

### Result :

```python
fig, axs = plt.subplots(figsize = (20, 20), nrows=1, ncols=2)
axs[0].imshow(detect_mask(img1)[:, :, ::-1])
axs[1].imshow(detect_mask(img2)[:, :, ::-1])
```

![download (1)](https://user-images.githubusercontent.com/91777895/147311871-d6b86cb4-b899-487f-b8ab-c11de2c84918.png)

