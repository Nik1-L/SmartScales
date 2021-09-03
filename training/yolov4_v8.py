# -*- coding: utf-8 -*-
"""
Обучение модели
"""

!git clone https://github.com/AlexeyAB/darknet

# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

!/usr/local/cuda/bin/nvcc --version

!make

def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
#   %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)

def download(path):
  from google.colab import files
  files.download(path)

# %cd ..
from google.colab import drive
drive.mount('/content/gdrive')

# %cd darknet

!cp /content/gdrive/MyDrive/yolov4/yolov4-obj.cfg ./cfg

!cp /content/gdrive/MyDrive/yolov4/obj.names ./data
!cp /content/gdrive/MyDrive/yolov4/obj.data  ./data

!cp /content/gdrive/MyDrive/yolov4/obj.zip ../
!cp /content/gdrive/MyDrive/yolov4/test.zip ../

!ls /content/gdrive/MyDrive/yolov4

!unzip ../obj.zip -d data/
!unzip ../test.zip -d data/

mv -v /content/darknet/data/obj/Apple_Banana_Orange_Lemon_Carrot_Tomato_Potato_Broccoli/* /content/darknet/data/obj

rm -R /content/darknet/data/obj/Apple_Banana_Orange_Lemon_Carrot_Tomato_Potato_Broccoli

mv -v /content/darknet/data/test/Apple_Banana_Orange_Lemon_Carrot_Tomato_Potato_Broccoli/* /content/darknet/data/test

rm -R /content/darknet/data/test/Apple_Banana_Orange_Lemon_Carrot_Tomato_Potato_Broccoli

!cp /content/gdrive/MyDrive/yolov4/generate_train.py ./
!cp /content/gdrive/MyDrive/yolov4/generate_test.py ./

!python generate_train.py
!python generate_test.py

!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /content/gdrive/MyDrive/yolov4/backup/yolov4-obj_last.weights -dont_show -map

!./darknet detector test data/obj.data cfg/yolov4-obj.cfg /content/gdrive/MyDrive/yolov4/yolov4-obj_last.weights /content/gdrive/MyDrive/yolov4/sem.jpg -thresh 0.1
imShow('predictions.jpg')
