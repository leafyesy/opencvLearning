import cv2
import numpy as np


def testFindConners(src, dst):
    img = cv2.pyrDown(src, cv2.IMREAD_UNCHANGED)
    #阈值操作 127为最低 255为最高
    ret, thresh = cv2.threshold(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(c)#找到最小的矩形区域
        box = cv2.boxPoints(rect)#转换为矩形的四个顶点
        box = np.int_(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (255, 0, 0), 1)

    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    cv2.imshow("findConner", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
