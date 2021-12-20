import cv2
import io  
import numpy as np
import random as rng 


rng.seed(12345)

# returns an image with bounding boxes draw around it 
def get_bounding_boxes(image):
    
    threshold = 100 

    # convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (3,3))

    canny_output = cv2.Canny(img_gray, threshold, threshold * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(image, contours_poly, i, color)
        cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)


    return image


# preprocess an image before sending it to the model
def blur_background(frame):

    # begin by converting image to grayscale
    img_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpening_kernel = np.array([
                        [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    image_blurred = cv2.medianBlur(frame, 5)
    image_sharpened = cv2.filter2D(src=image_blurred, ddepth=-1, kernel=sharpening_kernel)

    return image_sharpened