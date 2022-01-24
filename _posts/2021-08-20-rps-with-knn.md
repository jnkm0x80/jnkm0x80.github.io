---
layout: splash
title: "Rock Paper Scissors with KNN Algorithm"
---

<br>

<center><h1>Rock Paper Scissors with KNN Algorithm</h1></center>

## Libraries Used

|    Library     | URL                        |
|:--------------:|:-------------------------- |
|   **OpenCV**   | https://opencv.org         |
| **Matplotlib** | https://matplotlib.org     |
| **MediaPipe** | https://google.github.io/mediapipe |
|   **NumPy**    | https://numpy.org          |

```python
!pip install mediapipe --quiet

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
```

## Loading the Models from Libraries

### MediaPipe Hand Detection Model

```python
mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hand.Hands(
  max_num_hands = 3,
  min_detection_confidence = 0.7,
  min_tracking_confidence = 0.7
)
```

### OpenCV KNN Algorithm Model

Dataset: https://github.com/ntu-rris/google-mediapipe

```python
f = np.genfromtxt('gesture_train.csv', delimiter = ',')
angle = f[:, :-1].astype(np.float32)
label = f[:, -1].astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)
```
Output: True

## Sample Image

```python
img = cv2.imread('sample.jpg')
plt.figure(figsize = (20, 20))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

![download](https://user-images.githubusercontent.com/91777895/150855941-d627198e-451f-4abc-bb49-a80b22e9da5a.png)

### Processing Image

```python
img = cv2.flip(img, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = hands.process(img)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

## Applying KNN Algorithm for Gesture Detection

```python
rps = {0: 'rock', 5: 'paper', 6: 'paper', 1: 'scissors', 9: 'scissors'}

if result.multi_hand_landmarks is not None :

  for res in result.multi_hand_landmarks :

    joint = np.zeros((21, 3))
    for j, lm in enumerate(res.landmark) :
      joint[j] = [lm.x, lm.y, lm.z]

    v1 = joint[list(range(0, 20)), :] 
    v2 = joint[list(range(1, 21)), :] 
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[i for i in range(0, 19) if (i + 1) % 4 != 0], :],
                                v[[i for i in range(1, 20) if i % 4 != 0], :])) 
    angle = np.degrees(angle) 

    data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(data, 3)
    idx = int(results[0][0])

    if idx in rps.keys():
      cv2.putText(img, text = rps[idx].upper(), 
                  org = (int(res.landmark[0].x * img.shape[1]), 
                         int(res.landmark[0].y * img.shape[0])), 
                  fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, 
                  color = (0, 255, 0), thickness = 2)

    mp_draw.draw_landmarks(img, res, mp_hand.HAND_CONNECTIONS)
```

## Result

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (20, 20))
plt.imshow(img)
```

![download (1)](https://user-images.githubusercontent.com/91777895/150856121-c2e64bb9-7d25-4ec6-8c3b-812549dc4f51.png)

