import cv2
import numpy as np


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)  # 中值模糊
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, edgeKsize)  # 拉普拉斯

    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)

    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


def filter2D(src, dst):
    # 对感兴趣的像素来说,新像素是用当前像素乘9,然后减去8个临近像素值.如果像素间有差距,这个差距就会被放大
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    cv2.filter2D(src, -1, kernel, dst)


class VConvolutionFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


# 锐化边缘
class SharpenFilter(object):
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


# 转换边缘为白色,非边缘为黑色
class FindEdgesFilter(object):
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


# 模糊滤波器3
class BlurFilter(object):
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


# ridge或者浮雕(embossed)效果
class EmbossFilter(object):
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 0, 1],
                           [0, 1, 2]])
        self.impl = VConvolutionFilter(kernel)
