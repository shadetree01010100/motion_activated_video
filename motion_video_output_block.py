import numpy
import cv2
from threading import Event
from time import sleep

from nio import GeneratorBlock
from nio.signal.base import Signal
from nio.util.threading import spawn
from nio.properties import StringProperty, BoolProperty, VersionProperty, \
    IntProperty


class MotionVideoOutput(GeneratorBlock):
    """
    Open video source and output raw frames (numpy arrays)
    """

    source = StringProperty(title='Video Source', default='', allow_none=True)
    openOnStart = BoolProperty(title='Open source on start',
                               default=True,
                               visible=False)
    grayscale = BoolProperty(title='Convert to grayscale', default=False)
    frame_rate = IntProperty(title='Frames per second', default=0)
    non_motion_timer = 36
    min_upload_seconds = IntProperty(title='Minimum upload seconds', default=5)
    min_motion_frames = 12
    resize_width = 500
    min_area = 500
    delta_thresh = 5

    version = VersionProperty("0.0.1")

    def __init__(self):
        super().__init__()
        self._is_broadcasting = Event()
        self._thread = None

    def configure(self, context):
        super().configure(context)
        self.camera = None
        if self.openOnStart():
            self._openSource()

    def start(self):
        super().start()
        if self._is_broadcasting.is_set():
            self._thread = spawn(self._run)

    def stop(self):
        try:
            self._is_broadcasting.clear()
            self.logger.debug('Halting VideoInput thread')
            self.camera.release()
            self._thread.join()
        except:
            self.logger.exception('Exception while halting VideoInput')
        # cv2.destroyAllWindows()
        super().stop()

    def _run(self):
        non_motion_timer = self.non_motion_timer
        stream_video = False
        motion_counter = 0
        (h, w) = (None, None)
        avg = None
        while self._is_broadcasting.is_set():
            (grabbed, frame) = self.camera.read()
            motion_detected = False
            if self.frame_rate():
                sleep(1 / self.frame_rate())
            if not grabbed:
                self._is_broadcasting.clear()
                self.logger.exception('Failed to grab frame')
            else:
                # frame = imutils.resize(frame, width=self.resize_width)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # if the average frame is None, initialize it
                if avg is None:
                    print("[INFO] starting background model...")
                    avg = gray.copy().astype("float")
                    # frame.truncate(0)
                    continue
                cv2.accumulateWeighted(gray, avg, 0.5)
                frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

                # threshold the delta image, dilate the thresholded image to fill
                # in holes, then find contours on thresholded image
                thresh = cv2.threshold(frameDelta, self.delta_thresh, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                _, cnts, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # loop over the contours
                for c in cnts:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < self.min_area:
                        continue

                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w1, h1) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w1, y + h1), (0, 255, 0), 2)
                    motion_detected = True

                if motion_detected:

                    # increment the motion counter
                    motion_counter += 1

                    # check to see if the number of frames with motion is high enough
                    if motion_counter >= self.min_motion_frames:
                        stream_video = True
                        non_motion_timer = self.non_motion_timer
                        self.notify_signals([Signal({'frame': frame})])

                else:  # TODO: implement a max recording time
                    print("[DEBUG] no motion")
                    if stream_video is True and non_motion_timer > 0:
                        non_motion_timer -= 1
                        # print("[DEBUG] first else and timer: " + str(non_motion_timer))
                        self.notify_signals([Signal({'frame': frame})])
                    else:
                        print("[DEBUG] hit else")
                        motion_counter = 0
                        stream_video = False
                        non_motion_timer = self.non_motion_timer


    def _openSource(self):
        """ With no source, use attached camera """
        source = self.source()
        try:
            if not source:
                self.camera = cv2.VideoCapture(0)
            else:
                self.camera = cv2.VideoCapture(source)
            self._is_broadcasting.set()
        except:
            self._is_broadcasting.clear()
        self.logger.debug('Opening source: {}'.format(source if source else 'Local Camera'))
