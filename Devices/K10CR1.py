import numpy as np

from Devices.Templates import WAVEPLATE


class K10CR1(WAVEPLATE):

    def __init__(self, address, zero_pos):
        from pylablib.devices import Thorlabs
        self.address = address
        self.zero_pos = zero_pos
        print("Looking for Motor with id={}".format(address))
        for i in range(20):
            try:
                conn = {
                    "port": "/dev/ttyUSB{}".format(i),
                    "baudrate": 115200,
                    "rtscts": True
                }
                self.stage = Thorlabs.KinesisMotor(("serial", conn),
                                                   scale="stage")
                if (self.stage.get_device_info()[0] == int(address)):
                    print("Found Motor with id={}".format(address))
                    break
            except:
                print("No Device at /dev/ttyUSB{}".format(i))

    def move_to(self, pos, block=True):
        self.stage.move_to(pos + self.zero_pos)
        if block:
            self.wait_move()

    def wait_move(self):
        self.stage.wait_move()

    def stop(self):
        self.stage.stop()

    def home(self, block=True):
        print("Homing")
        self.stage.home(sync=False)
        if block:
            self.wait_home()
            print("Homing complete")

    def wait_home(self):
        self.stage.wait_for_home()

    def setup_jog(self, speed, limit=720):
        self.stage.setup_jog(step_size=limit, max_velocity=speed)

    def jog(self, direction="+"):
        self.stage.jog(direction, kind="builtin")
