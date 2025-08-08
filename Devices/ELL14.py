from Devices.Templates import WAVEPLATES

from pylablib.devices import Thorlabs
from serial.tools import list_ports


def normalize_angle(angle_degrees):
    """Normalize any angle to the range [-180, 180)."""
    normalized = (angle_degrees + 180) % 360 - 180
    return normalized


class ELL14(WAVEPLATES):
    def __init__(self, serial_number, zero_pos, hwp_angle0, qwp_angle0):
        self.serial_number = serial_number
        self.zero_pos = zero_pos
        self.hwp_angle0 = hwp_angle0
        self.qwp_angle0 = qwp_angle0
        print("Looking for Motor with serial={}".format(serial_number))
        port = ""
        for p in list_ports.comports():
            if serial_number == p.serial_number:
                port = p.device
                print("Found Motor")
                break
        if port == "":
            raise Exception("No Device with serial_number={}".format(serial_number))
        conn = {
            "port": port,
            "baudrate": 9600,
            # "rtscts": True,
        }
        self.stage = Thorlabs.ElliptecMotor(("serial", conn), scale="stage")
        self.addrs = self.stage.get_connected_addrs()

    def stop(self):
        pass

    def home(self):
        for addr in self.addrs:
            print("Homing WP {}".format(addr))
            self.stage.home(addr=addr)

    def jog_like_HWP(self, speed):
        raise Exception("ELL14 can not jog")

    def move_like_HWP(self, angle):
        for addr in self.addrs:
            self.stage.move_to(
                normalize_angle(self.zero_pos[addr] + self.hwp_angle0[addr] + angle),
                addr=addr,
            )

    def move_like_QWP(self, angle):
        for addr in self.addrs:
            self.stage.move_to(
                normalize_angle(self.zero_pos[addr] + self.qwp_angle0[addr] + angle),
                addr=addr,
            )

    def move_to(self, angle):
        for addr in self.addrs:
            self.stage.move_to(
                normalize_angle(self.zero_pos[addr] + angle),
                addr=addr,
            )


def list_serials():
    for p in list_ports.comports():
        if p.serial_number:
            print("{}->{}".format(p.device, p.serial_number))


list_serials()
