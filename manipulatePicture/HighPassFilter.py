import cv2
import numpy as np
from scipy import ndimage

kernel_33 = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])

kernel_55 = np.array([[-1, -1, -1, -1, -1],
                      [-1, 1, 2, 1, -1],
                      [-1, 2, 4, 2, -1],
                      [-1, 1, 2, 2, -1],
                      [-1, -1, -1, -1, -1]])

img = cv2.imread("../img/2.jpg", cv2.IMREAD_GRAYSCALE)

k3 = ndimage.convolve(img, kernel_33)
k5 = ndimage.convolve(img, kernel_55)

blurred = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - blurred

cv2.imshow("3*3", k3)
cv2.imshow("5*5", k5)

cv2.imshow("g_hpf", g_hpf)

cv2.waitKey(0)
cv2.destroyAllWindows()
