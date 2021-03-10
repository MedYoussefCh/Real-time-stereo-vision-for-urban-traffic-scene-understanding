# Real-time-stereo-vision-for-urban-traffic-scene-understanding

[As part of my formation in Ecole Centrale Casablanca, we were led to work in groups of two on this project, me and Amine Nait Charif in collaboration with the MAScIR foundation, supervised by Mr Omar Bourja (foundation MAScIR), Mr Hamd Ait Abdelali (foundation MAScIR) and Mr Khalid Dahi (Ecole Centrale Casablanca).]

## i. Stereoscopic survey : application to traffic 

- Stereo vision is one of the main researched domains of computer vision, and it can be used for different applications, among them, we can find extraction of the depth of a scene, estimating the distance between the camera and other objects and then estimate the distance between objects which we are going to see in this article. Stereoscopy is a computer vision discipline used for recording and representing stereoscopic images (3D) . It can create an illusion of depth using two pictures taken from slightly different angles. There are actually two possible ways to take stereoscopic pictures : first, by using special two-lens stereo cameras or systems with two single-lens cameras joined together, The 3D camera consists of two cameras of parallel optical axes and separated horizontally from each other by a small distance and these two cameras are combined together in a single frame in order to get a 3D image. The 3D camera is used to produce two stereoscopic pictures for a given object. The distance between the cameras and the object can be measured depending on the distance between the positions of the objects in both pictures, the focal lengths of both cameras and the distance as well. In this article, we provide a comprehensive survey of this continuously growing field of research, summarize the foremost commonly uses of stereoscopic cameras especially its applications to traffic, and discuss their benefits and limitations. In retrospect of what has been achieved so far.

## ii. Real Time vehicle Detection, Tracking, and Inter-vehicle Distance Estimation based on Stereo Vision and Deep Learning using YOLO v3. 

- In this study, we propose a robust real-time vehicle tracking and inter-vehicle distance estimation algorithm based on stereovision. Traffic images are captured by a stereoscopic system installed on the road, and then we detect moving vehicles with the YOLO V3 Deep Neural Network algorithm. Thus, the real-time video goes through an algorithm for stereoscopy-based measurement in order to estimate distance between detected vehicles. However, detecting the real-time objects have always been a challenging task because of occlusion, scale, illumination etc. Thus, many convolutional neural network models based on object detection were developed in recent years. But they cannot be used for real-time object analysis because of slow speed of recognition. The model which is performing excellent currently is the unified object detection model which is You Only Look Once (YOLO). But in our experiment, we have found that despite of having a very good detection precision, YOLO still has some limitations. YOLO processes every image separately even in a continuous video or frames. Because of this much important identification can be lost. So, after the vehicle detection and tracking, inter-vehicle distance estimation is done. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#### Make sure python 3 is installed with opencv-contrib and numpy pip installed
#### Make sure the folder contains all of these files/folders:
- coco.names 
- yolov3.cfg 
- yolov3.weights
- yoloModule.py 
- denseDisparity.py 
- run.py 
- STEREOVISION DATA (folder containing left-images and right-images) 
#### in run.py edit the "PATH_TO_DATASET" variable value to the path containing the stereo images
#### execute run.py 
#### press "x" at any point to exit and shut down all windows

[Artificial Intelligence - Computer stereo vision - Image processing - Deep Learning - YOLO v3 - Python (OpenCV, Tensorflow, Keras, etc.)]

