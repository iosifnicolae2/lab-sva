# import cv2
# import numpy as np
#
# if __name__ == "__main__" :
#     img = cv2.imread( "opencv_screenshot.jpg" )
#     # pen t ru a a p l i c a f i l t r u l , schimbam r e p r e z e n t a r e a i m a g i n i i
#     # din numere i n t r e g i i n numere r e a l e
#     # img = img.astype (np.float32 ) / 255.0
#
#     # kernel_size = 3
#     # kernel = np.ones ( ( kernel_size , kernel_size ) , dtype=np . float )
#     # kernel = ( 1 / ( kernel_size * kernel_size ) ) * kernel
#     #
#     # # f i l t r a r e im a gine
#     # img_filtered = cv2.filter2D( img , -1, kernel )
#     # img_filtered2 = cv2.medianBlur( img , kernel_size)
#
#     sobelx = cv2.Sobel(img,cv2.CV_64F, dx=4, dy=0,ksize=5)
#
#     cv2.imshow('image', sobelx)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('FaceTime_2019-10-17 19-44-59@2x.png',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


plt.show()

# blur = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow('image', blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()