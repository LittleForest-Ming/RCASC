# Crop rows Detection Inference Pytorch
Example scripts for the detection of rowss using the [RASC models](https://github.com/) in Pytorch.

![!crop rows](https://github.com/xiapming123/RCASC/blob/main/roadmap.png)
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

 * **Input**: RGB image of size 1920 x 1080 pixels.
 * **Output**: Keypoints for a maximum of 4 rows (core areasleft-most row, left row, right lane, and right-most lane).
 
# Examples

 * **Image inference**:
 
 ```
 python imageDetection.py 
 ```
 
  * **Webcam inference**:
 
 ```
 python webcamDetection.py
 ```
 
  * **Video inference**:
 
 ```
 python videoLaneDetection.py
 ```
 
 # [Inference video Example](https://pan.baidu.com/s/1yrRWAZCg32CGp2oNKnScKw?pwd=ey57) 
 ![!Crop rows detection on video](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/blob/main/doc/img/laneDetection.gif)
 
 Original video: https://pan.baidu.com/s/1yrRWAZCg32CGp2oNKnScKw?pwd=ey57 -- (by Zhiming Guo)
 
