import cv2
import numpy as np

if __name__ == '__main__':

    filename = 'chessboard.jpeg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.001*dst.max()]=[0,200,255]

    # cv2.imshow('dst',img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    cv2.drawKeypoints( img , kp , img )

    # fast = cv2.FastFeatureDetector.create(10)
    # kp = fast.detect(gray)