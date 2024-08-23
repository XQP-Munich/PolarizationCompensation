import numpy as np


class Waveplates():

    def __init__(self, waveplates, hwp_angle0, hwp_speed, qwp_angle0):
        self.waveplates = waveplates
        self.hwp_angle0 = hwp_angle0
        self.hwp_speed = hwp_speed
        self.qwp_angle0 = qwp_angle0

    def jog_like_HWP(self, speed):
        self.move_to(self.hwp_angle0)
        print("Setting jog params")
        for i, wp in enumerate(self.waveplates):
            wp.setup_jog(speed=self.hwp_speed[i] * speed)
        print("Start jog")
        for wp in self.waveplates:
            wp.jog()

    def move_like_QWP(self, angle):
        self.move_to([x + angle for x in self.qwp_angle0])

    def stop(self):
        print("Stopping waveplates")
        for wp in self.waveplates:
            wp.stop()

    def move_to(self, pos):
        print("Moving Waveplates to {}".format(pos))
        if isinstance(pos, list) or isinstance(pos, np.ndarray):
            for i, wp in enumerate(self.waveplates):
                wp.move_to(pos[i], block=False)
        else:
            for wp in self.waveplates:
                wp.move_to(pos, block=False)

        for wp in self.waveplates:
            wp.wait_move()
        print("done")

    def home(self):
        for wp in self.waveplates:
            wp.home(block=False)
        print("Waiting for home")
        for wp in self.waveplates:
            wp.wait_home()
        print("Homing complete")
