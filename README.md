# Real-time-stereo-vision-for-urban-traffic-scene-understanding

*As part of my formation in Ecole Centrale Casablanca, we were led to work in groups of two on this project, me and Amine Nait Charif in collaboration with the MAScIR foundation, supervised by Mr Omar Bourja (Embedded Systems Project Manager - MAScIR), Mr Hamd Ait Abdelali (scientific researcher - MAScIR), Mr Hatim Derrouz (scientific researcher - MAScIR) and Mr Khalid Dahi (scientific researcher - Ecole Centrale Casablanca).

## i. Stereoscopic survey : application to traffic 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Stereo vision is one of the main researched domains of computer vision, and it can be used for different applications, among them, we can find extraction of the depth of a scene, estimating the distance between the camera and other objects and then estimate the distance between objects which we are going to see in this study. Stereoscopy is a computer vision discipline used for recording and representing stereoscopic images (3D) . It can create an illusion of depth using two pictures taken from slightly different angles. There are actually two possible ways to take stereoscopic pictures : first, by using special two-lens stereo cameras or systems with two single-lens cameras joined together, The 3D camera consists of two cameras of parallel optical axes and separated horizontally from each other by a small distance and these two cameras are combined together in a single frame in order to get a 3D image. The 3D camera is used to produce two stereoscopic pictures for a given object. The distance between the cameras and the object can be measured depending on the distance between the positions of the objects in both pictures, the focal lengths of both cameras and the distance as well. In this study, we provide a comprehensive survey of this continuously growing field of research, summarize the foremost commonly uses of stereoscopic cameras especially its applications to traffic, and discuss their benefits and limitations. In retrospect of what has been achieved so far.

## ii. Real Time vehicle Detection, Tracking, and Inter-vehicle Distance Estimation based on Stereo Vision and Deep Learning using YOLO v3. 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- In this study, we propose a robust real-time vehicle tracking and inter-vehicle distance estimation algorithm based on stereovision. Traffic images are captured by a stereoscopic system installed on the road, and then we detect moving vehicles with the YOLO V3 Deep Neural Network algorithm. Thus, the real-time video goes through an algorithm for stereoscopy-based measurement in order to estimate distance between detected vehicles. However, detecting the real-time objects have always been a challenging task because of occlusion, scale, illumination etc. Thus, many convolutional neural network models based on object detection were developed in recent years. But they cannot be used for real-time object analysis because of slow speed of recognition. The model which is performing excellent currently is the unified object detection model which is You Only Look Once (YOLO). But in our experiment, we have found that despite of having a very good detection precision, YOLO still has some limitations. YOLO processes every image separately even in a continuous video or frames. Because of this much important identification can be lost. So, after the vehicle detection and tracking, inter-vehicle distance estimation is done. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#### Make sure python 3 is installed with opencv-contrib and numpy pip installed
#### Make sure the folder contains all of these files/folders:
- coco.names 
- yolov3.cfg [https://centralecasablanca-my.sharepoint.com/:u:/g/personal/mohamed_chouhaidi_centrale-casablanca_ma/EY7-LZdqhdFPnJ6951aQJ_sBKmuE-_jQQX0Jo2M1PSHesg?e=zHTZz3]
- yolov3.weights [https://centralecasablanca-my.sharepoint.com/:u:/g/personal/mohamed_chouhaidi_centrale-casablanca_ma/EQPzTbjZ90hJlfygeHMQ6psBwwBCtWGCSmxlobV34mdjiA?e=QZxfDt]
- yoloModule.py [https://centralecasablanca-my.sharepoint.com/:u:/g/personal/mohamed_chouhaidi_centrale-casablanca_ma/ETAngl8p2PpJq56miuu-Q1QBM0R4t8Toa86hlL_btBs2XA?e=JXe6Lp]
- denseDisparity.py 
- main_Disparity.py
- STEREOVISION DATA (folder containing left-images 1 and right-images 2) [https://centralecasablanca-my.sharepoint.com/:f:/g/personal/mohamed_chouhaidi_centrale-casablanca_ma/Eoahs_F2Rv1JtSF4VwiiRMYB6bJpQrkJcYBC-whYmMin0w?e=wVjgEW]
#### in main_Disparity.py  edit the "PATH_TO_DATASET" variable value to the path containing the stereo images
#### execute main_Disparity.py 
#### press "x" at any point to exit and shut down all windows

## References : 
- https://github.com/tobybreckon/stereo-disparity
- https://pjreddie.com/media/files/papers/YOLOv3.pdf
- https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
- https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.py
- https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/LICENSE
- https://pjreddie.com/media/files/yolov3.weights
- https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
- https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true

[Artificial Intelligence - Computer stereo vision - Image processing - Deep Learning - YOLO v3 - Python (OpenCV, Tensorflow, Keras, etc.)]

