from manipulatePicture import Filter
import manipulatePicture.ConnersTest as ct
import numpy as np
import os, cv2

path = os.getcwd()

src = cv2.imread(path + r"/img/2.jpg", cv2.IMREAD_COLOR)
dst = np.zeros(src.shape, dtype=np.uint8)

# test strokeEdges 边缘描绘
# Filter.strokeEdges(src, dst, 7, 5)

# test fliter2D
# Filter.filter2D(src, dst)

# temPath = path + r"/img/canny.jpg"
# cv2.imwrite(temPath, cv2.Canny(src, 200, 300))
# cv2.imshow("canny", cv2.imread(temPath))

ct.testFindConners(src, dst)


# cv2.imshow("dst", dst)
# cv2.waitKey()
# cv2.destroyAllWindows()
