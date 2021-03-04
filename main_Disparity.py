import os
import cv2
import denseDisparity as dd
import yoloModule as yolo
import numpy as np
import math

master_path_to_dataset = "C:\\Users\\Youssef\\Documents\\STEREOVISION_DATA"

directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

def prep(img):
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    planes = cv2.split(lab)
    clahe= cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
    planes[0] = clahe.apply(planes[0])
    lab = cv2.merge(planes)
    img = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    return img
def drawPred(image, class_name, left, top, right, bottom, distance, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)


    label = '%s:%.2f' % (class_name, distance)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


def run_program():
    full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left)
    full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right)
    saveFolderColour = os.path.join(master_path_to_dataset,"dist")
    saveFolderMaps = os.path.join(master_path_to_dataset,"maps")
    left_file_list = sorted(os.listdir(full_path_directory_left))

    for filename_left in left_file_list:
        filename_right = filename_left.replace("_L","_R")
        full_path_filename_left = os.path.join(full_path_directory_left,filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

        if (".png" in filename_left) and (os.path.isfile(full_path_filename_right)):

            imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

            imgL = prep(imgL)

            imgR = prep(imgR)
            print("-- files loaded successfully")
            print()


            l_img_g = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
            r_img_g = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)



            l_img_g = np.power(l_img_g, 0.75).astype('uint8')
            r_img_g = np.power(r_img_g, 0.75).astype('uint8')


            #find objects
            objects = yolo.detection(imgL[0:390])
            #create disparitymap
            disparityMap = dd.disparity_map(l_img_g,r_img_g)
            distances = []
            #find distances
            for (_, box) in objects:
                distances.append(dd.distance_from_disparity(disparityMap,box))

            if objects == []:
                closest_obj = "nothing"
                closest_dist = 0
            else:
                min_dist_index = distances.index(min(distances))
                closest_obj = objects[min_dist_index][0]
                closest_dist = distances[min_dist_index]

            print(filename_left)
            print("{0} : {1} {2}m".format(filename_right,closest_obj,closest_dist))
            #draw boxes
            for i in range(len(objects)):
                left = objects[i][1][0]
                top = objects[i][1][1]
                right = left + objects[i][1][2]
                bottom = top + objects[i][1][3]

                drawPred(imgL,objects[i][0],left,top,right,bottom,distances[i], (255, 178, 50))
            cv2.namedWindow("left image", cv2.WINDOW_NORMAL)
            cv2.imshow("left image",imgL)
            cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
            cv2.imshow("disparity", (disparityMap * (256. / 128)).astype(np.uint8))

            key = cv2.waitKey(1)
            if key == ord("x"):
                break
    cv2.destroyAllWindows()



run_program()