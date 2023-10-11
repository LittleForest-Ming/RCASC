# Crop rows Detection Inference Pytorch
Example scripts for the detection of rowss using the [RASC models](https://github.com/) in Pytorch.

![!Ultra fast lane detection](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/blob/main/doc/img/.jpg)
image-Source: RASC/doc

# Requirements

 * **OpenCV**, **Scikit-learn** and **pytorch**. 
 
# Installation
```
pip install -r requirements

```
**Pytorch:** Check the [Pytorch website](https://pytorch.org/) to find the best method to install Pytorch in your computer.

# Pretrained model
Download the pretrained model from the [Baidu Netdisk](https://pan.baidu.com/s/1PaIxvx_twgHaCC6OjF3ivA) extraction code: xm1m 


# Ultra fast lane detection - TuSimple([link](https://github.com/cfzd/Ultra-Fast-Lane-Detection))

 * **Input**: RGB image of size 1920 x 1080 pixels.
 * **Output**: Keypoints for a maximum of 4 rows (core areasleft-most row, left row, right lane, and right-most lane).
 
# Examples

 * **Image inference**:
 
 ```
 python imageLaneDetection.py 
 ```
 
  * **Webcam inference**:
 
 ```
 python webcamLaneDetection.py
 ```
 
  * **Video inference**:
 
 ```
 python videoLaneDetection.py
 ```
 
 # [Inference video Example](https://youtu.be/0Owf6gef1Ew) 
 ![!Ultrafast lane detection on video](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/blob/main/doc/img/laneDetection.gif)
 
 Original video: https://youtu.be/2CIxM7x-Clc (by Yunfei Guo)
 
