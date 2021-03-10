import cv2
import numpy as np



def disparity_map(l_img_g, r_img_g, WLS=False):

    max_disparity = 128
    left_matcher = cv2.StereoSGBM_create(0, max_disparity, 21)

    if WLS:
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        lmbda = 80000
        sigma = 1.2
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

    disparityL = left_matcher.compute(l_img_g, r_img_g)
    if WLS:
        disparityR = right_matcher.compute(l_img_g,r_img_g)
        disparity = wls_filter.filter(disparityL,l_img_g,None,disparityR)
    else:
        disparity = disparityL

    dispNoiseFilter = 5
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    disparity_scaled = disparity_scaled[0:390]

    return disparity_scaled

def distance_from_disparity(disparity, box): #disparity = disparity map, box = object detected via yolo
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    dispH,dispW = disparity.shape
    feature_sub_array = disparity[max(top,0):min(top+height+1,dispH),max(136,left):min(left+width+1,dispW)]
    average_feature_disparity = np.nansum(feature_sub_array) / np.count_nonzero(feature_sub_array)
    f = 399.9745178222656
    b = 0.2090607502
    if average_feature_disparity != 0:
        return (f*b)/average_feature_disparity
    else:
        return 0



