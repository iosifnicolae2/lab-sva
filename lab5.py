import cv2
import numpy

WIDTH = 500
HEIGHT = 500
MIDDLE = int(HEIGHT/2)

img = numpy.zeros([WIDTH, HEIGHT])


for i in range(0, WIDTH):
    img[MIDDLE,i] = 255 # y,x

cv2.imwrite('original.jpg', img)

alfa = numpy.pi / 2
# alfa = 0
tx = 0
ty = 100

rot = numpy.array([
    [numpy.cos(alfa), -numpy.sin(alfa), tx],
    [numpy.sin(alfa), numpy.cos(alfa), ty],
    [0,0,1],
])

rotated_img = numpy.matmul(
        rot,
        point_matrix,
)

for i in range(0, WIDTH):
    px = i
    py = MIDDLE
    point_matrix = numpy.array([[py], [px], [1]])

    new_point = numpy.matmul(
        rot,
        point_matrix,
    )
    img[py, px] = 0  # punctul vechi
    new_point_x = int(new_point[0][0])
    new_point_y = int(new_point[1][0])
    img[new_point_x, new_point_y] = 255


cv2.imwrite('rotated.jpg', img)


if __name__ == '__main__':
    pass





