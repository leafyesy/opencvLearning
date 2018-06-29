import cv2
from manipulatePicture.managers import WindowManager, CaptureManager
import manipulatePicture.Filter as filters


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = filters.EmbossFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreate:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            self._curveFilter.impl.apply(frame,frame)
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keyword):
        if keyword == 32:  # space
            self._captureManager.writeImage('screenshot.png')
        elif keyword == 9:
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keyword == 27:  # escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()
