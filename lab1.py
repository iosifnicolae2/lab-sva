import cv2
import numpy


IMAGE_FILE = 'image.jpg'


def produs(sir):
    array = numpy.array(sir)
    return numpy.prod(array)

    # or
    # p = 1
    # for i in sir:
    #     p *= i
    # return p


def invers(sir):
    # v = []
    # sz = len(sir)
    # for i in range(0, sz):
    #     v.append(sir[sz-i-1])
    # return v

    array = numpy.array(sir)
    return numpy.flip(array)


class ImageProcessing:
    def __init__(self):
        img = None

    def read(self, file):
        self.img = cv2.imread(file, cv2.IMREAD_COLOR)

    def show(self):
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_to_grayscale(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)


def matrix_operations():
    a = numpy.matrix([[2, -1, 1], [5, 2, -3], [2, 1, -1]])
    b = numpy.matrix([[3], [1], [2]])
    x = numpy.linalg.solve(a, b)
    print("a: {}".format(a))
    print("b: {}".format(b))
    print("x: {}".format(x))


if __name__ == '__main__':
    # Python
    vect = [2,3,4]
    a = produs(vect)
    print(a)

    b = invers(vect)
    print(b)

    # OpenCV
    img_process = ImageProcessing()
    img_process.read(IMAGE_FILE)
    img_process.show()
    img_process.convert_to_grayscale()
    img_process.show()

    # Numpy
    matrix_operations()
