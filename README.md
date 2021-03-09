# Real-time-stereo-vision-for-urban-traffic-scene-understanding
- 1. Stereoscopic survey : application to traffic 
- 2. Real Time vehicle Detection, Tracking, and Inter-vehicle Distance Estimation based on Stereo Vision and Deep Learning using YOLO v3. (Artificial Intelligence - Computer stereo vision - Image processing - Deep Learning - YOLO v3 - Python - OpenCV - Tensorflow - Keras)

In this study, we propose a robust real-time vehicle tracking and inter-vehicle distance estimation algorithm based on stereovision. Traffic images are captured by a stereoscopic system installed on the road, and then we detect moving vehicles with the YOLO V3 Deep Neural Network algorithm. Thus, the real-time video goes through an algorithm for stereoscopy-based measurement in order to estimate distance between detected vehicles. However, detecting the real-time objects have always been a challenging task because of occlusion, scale, illumination etc. Thus, many convolutional neural network models based on object detection were developed in recent years. But they cannot be used for real-time object analysis because of slow speed of recognition. The model which is performing excellent currently is the unified object detection model which is You Only Look Once (YOLO). But in our experiment, we have found that despite of having a very good detection precision, YOLO still has some limitations. YOLO processes every image separately even in a continuous video or frames. Because of this much important identification can be lost. So, after the vehicle detection and tracking, inter-vehicle distance estimation is done. 
