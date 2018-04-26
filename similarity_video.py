from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np
from keras.layers import merge, Input


# from imagenet_utils import decode_predictions

model = VGG19(include_top=False,weights="imagenet")

image_input = Input(shape=(224,224,3))

model = VGG19(include_top=False,input_tensor=image_input,weights="imagenet")

model.summary()

import argparse
import cv2
import numpy as np
import os
import random
import sys
import scipy

import cv2, pafy

url1 = "https://www.youtube.com/watch?v=vx2u5uUu3DE"
url2 = "https://www.youtube.com/watch?v=orw3G3jLYeM"

videoPafy1 = pafy.new(url1)
best1 = videoPafy1.getbest(preftype="webm")

videoPafy2 = pafy.new(url2)
best2 = videoPafy2.getbest(preftype="webm")

cap1 = cv2.VideoCapture(best1.url)
cap2 = cv2.VideoCapture(best2.url)
similar = []

if (cap1.isOpened()) and (cap2.isOpened())  :
    print("Camera OK")
else:
    cap1.open() and cap2.open() 

while (True):
    ret1, original1 = cap1.read()
    ret2, original2 = cap2.read()
    image1 = cv2.resize(original1, (224, 224))
    image2 = cv2.resize(original2, (224, 224))
    #image = image_utils.load_img(frame, target_size=(224, 224))
    image1= image.img_to_array(image1)
    image2= image.img_to_array(image2)
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    image1 = preprocess_input(image1)
    image2 = preprocess_input(image2)
    pred1 = model.predict(image1)
    pred2 = model.predict(image2)
    vgg16_feature_1 = np.array(pred1)
    vgg16_feature_1 = vgg16_feature_1.flatten()
    vgg16_feature_2 = np.array(pred2)
    vgg16_feature_2 = vgg16_feature_2.flatten()
    score = scipy.spatial.distance.cosine(u=vgg16_feature_1,v=vgg16_feature_2)
    print(score)
    similar.append(score)
    cv2.imshow("video 1", original1)
    cv2.imshow("video 2", original2)
    if score==0:
        cv2.imwrite("C:\\Users\\User.1\\Documents\\Gaurav\\DL\\similar images\\frame1.jpg",image1)
        cv2.imwrite("C:\\Users\\User.1\\Documents\\Gaurav\\DL\\similar images\\frame2.jpg",image2)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap1.release()
cap2.release()
frame = None
cv2.destroyAllWindows()
sys.exit()