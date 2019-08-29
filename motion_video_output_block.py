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

    frame_rate = IntProperty(title='Frames per second', default=10)
    non_motion_timer = IntProperty(title='Release Frames', default=36)
    min_motion_frames = IntProperty(title='Trigger Frames', default=12)
    min_area = IntProperty(title='Countour Area', default=500)
    delta_thresh = IntProperty(title='Delta Threshold', default=5)

    version = VersionProperty("0.0.1")

    def __init__(self):
        super().__init__()
        self._is_broadcasting = Event()
        self._thread = None
        self._camera = None
        self._active = False

    def start(self):
        super().start()
        spawn(self._openSource)

    def stop(self):
        self.logger.debug('Halting thread')
        try:
            self._is_broadcasting.clear()
            self._thread.join()
            self._camera.release()
        except:
            self.logger.exception('Exception while halting VideoInput')
        super().stop()

    def _run(self):
        non_motion_timer = self.non_motion_timer()
        stream_video = False
        motion_counter = 0
        avg = None
        try:
            while self._is_broadcasting.is_set():
                # control frame rate
                if self.frame_rate():
                    sleep(1 / self.frame_rate())
                # grab a frame
                success, frame = self._camera.read()
                if not success:
                    self.logger.exception('Failed to grab frame')
                    break
                # detect motion
                motion_detected = False
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                # if the average frame is None, initialize it
                if avg is None:
                    self.logger.debug("starting background model...")
                    avg = gray.copy().astype("float")
                    continue
                cv2.accumulateWeighted(gray, avg, 0.5)
                frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
                # threshold the delta image, dilate the thresholded image to fill
                # in holes, then find contours on thresholded image
                thresh = cv2.threshold(frameDelta, self.delta_thresh(), 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                _, cnts, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # loop over the contours
                for c in cnts:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < self.min_area():
                        continue

                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w1, h1) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w1, y + h1), (0, 255, 0), 2)
                    motion_detected = True
                if motion_detected:
                    motion_counter += 1
                    # check to see if the number of frames with motion is high enough
                    if motion_counter >= self.min_motion_frames():
                        if not stream_video:
                            self.logger.debug('Motion Detected!')
                        stream_video = True
                        non_motion_timer = self.non_motion_timer()
                        self.notify_signals([Signal({'frame': frame})])

                else:
                    if stream_video is True and non_motion_timer > 0:
                        non_motion_timer -= 1
                        self.notify_signals([Signal({'frame': frame})])
                    else:
                        if stream_video:
                            self.logger.debug('Stream Stopped')
                        motion_counter = 0
                        stream_video = False
                        non_motion_timer = self.non_motion_timer()
            if self._is_broadcasting.is_set():
                # loop was broken, respawn!
                self._is_broadcasting.clear()
                self._camera = None
                self._thread = spawn(self._openSource)
        except:
            self.logger.exception('Error in video read loop!')
            self._is_broadcasting.clear()
            self._camera = None
            self._thread = spawn(self._openSource)

    def _openSource(self):
        while True:
            self.logger.debug('Opening camera')
            try:
                cam = cv2.VideoCapture(0)
                if cam.isOpened():
                    self.logger.debug('Got Camera!')
                    self._camera = cam
                    self._is_broadcasting.set()
                    break
                self.logger.warning('Failed to open camera, retrying...')
            except:
                self._is_broadcasting.clear()
            sleep(1)
        self._thread = spawn(self._run)
