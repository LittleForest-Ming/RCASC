# Crop rows Detection Inference Pytorch
Example of an image using the [RCASC models](https://github.com/) in Pytorch to detect a corn crop row (adaptive extraction of the core region).

![!crop rows](https://github.com/xiapming123/RCASC/blob/main/roadmap.png)
image-Source: RASC/doc. 
For ease of use, we have expanded the row-anchor version of the corn crop row detection code and are constantly improving it.

# Requirements

 * **OpenCV**, **Scikit-learn** and **pytorch**. 
 
# Installation
```
pip install -r requirements

```
**Pytorch:** Check the [Pytorch website](https://pytorch.org/) to find the best method to install Pytorch in your computer.

# Pretrained model
Download the pretrained model from the [Baidu Netdisk](https://pan.baidu.com/s/1PaIxvx_twgHaCC6OjF3ivA) .extraction code: xm1m 

 * **Input**: RGB image of size 1920 x 1080 pixels.
 * **Output**: Keypoints for a maximum of 4 rows (core areas: left-most row, left row, right row, and right-most row).
 
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
 python videoDetection.py
 ```
 
 # [Inference video Example](https://pan.baidu.com/s/1yrRWAZCg32CGp2oNKnScKw?pwd=ey57) 
 ![!Crop rows detection on video](https://github.com/xiapming123/RCASC/blob/main/Video-Results/1%20-middle-original.gif)
 
 Original video: https://pan.baidu.com/s/1yrRWAZCg32CGp2oNKnScKw?pwd=ey57 -- (by Zhiming Guo)
 
