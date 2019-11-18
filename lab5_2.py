# V2
import numpy

points = numpy.array([
    [1, 2, 1],
    [2, 2, 1],
    [3, 2, 1],
    [4, 2, 1],
    [5, 2, 1],
])
print("points", points)
alfa = numpy.pi / 2
# alfa = 0
tx = 0
ty = 0

rot = numpy.array([
    [numpy.cos(alfa), -numpy.sin(alfa), tx],
    [numpy.sin(alfa), numpy.cos(alfa), ty],
    [0,0,1],
])
rotated_points =numpy.dot(rot, points.T)
print("rotated_points\n", rotated_points)

if __name__ == '__main__':
    pass