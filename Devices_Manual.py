import numpy as np
import Devices as dev

# File for device specific Declarations and Functions


# Class/Function definition
class Waveplate(dev.WAVEPLATE):

    def __init__(self, address, zero_pos, step_size):
        self.address = address
        self.zero_pos = zero_pos
        self.step_size = step_size

    def move_to(self, pos):
        input("[Waveplate] Please rotate Waveplate {} to: {}".format(
            self.address, pos))

    def jog(self, direction, speed):
        input("[Waveplate] Please start Waveplate {}"
              " jog in {} direction with speed {}°/s".format(
                  self.address, direction, speed))

    def stop(self):
        input("[Waveplate] Please stop Waveplate {} jog".format(self.address))


class Waveplates(dev.WAVEPLATES):

    def __init__(self, waveplates):
        self.waveplates = waveplates

    def jog_like_HWP(self, speed):
        input(
            "[Waveplates] Please start jog Waveplates like a single HWP with speed {}"
            .format(speed))


class Source(dev.SOURCE):

    def turn_on(self, pol):
        input("[Source] Please Turn on Polarisation: {}".format(pol))

    def turn_off(self):
        input("[Source] Please Turn off")


class Timestamp(dev.TIMESTAMP):

    def read(self, path):
        input("[Timestamp] Please start reading from timestamp to {}".format(
            path))

    def stop(self):
        input("[Timestamp] Please stop reading")

    def get_counts_per_second(self):
        return [[379305.0, 234931.0, 292022.0, 420890.0],
                [345783.0, 232888.0, 264730.3333333333, 408387.6666666667],
                [
                    320067.8333333333, 264639.1666666667, 252921.66666666666,
                    430680.0
                ], [285327.0, 286035.0, 246005.0, 423430.0],
                [
                    231410.3333333333, 287761.6666666667, 234864.16666666666,
                    376689.1666666667
                ], [206550.0, 303176.0, 246533.0, 352448.6666666667],
                [160145.0, 265083.0, 226156.0, 274461.0],
                [
                    157127.66666666666, 281849.6666666667, 261532.3333333333,
                    254097.0
                ], [186515.0, 332408.0, 338035.0, 270389.0],
                [186984.0, 308681.0, 351635.0, 228403.0],
                [
                    213841.66666666666, 301160.6666666667, 388515.3333333333,
                    211135.0
                ],
                [
                    249205.66666666666, 288847.6666666667, 417438.3333333333,
                    204872.33333333334
                ], [276009.0, 257417.0, 416495.0, 199441.0],
                [331226.6666666667, 251245.0, 446867.0, 230082.66666666666],
                [368113.0, 237763.0, 442224.3333333333, 261252.66666666666],
                [333269.0, 192096.0, 356256.0, 252503.0],
                [
                    334826.3333333333, 184838.0, 326256.3333333333,
                    279706.6666666667
                ],
                [
                    348770.0, 199034.66666666666, 315647.6666666667,
                    326534.3333333333
                ], [440721.0, 269873.0, 373698.0, 463838.0],
                [
                    415602.1111111111, 301213.55555555556, 354215.77777777775,
                    488385.22222222225
                ],
                [
                    387963.6666666667, 331267.3333333333, 336113.3333333333,
                    508022.0
                ], [389488.0, 401608.0, 368837.0, 560907.0],
                [
                    344026.3333333333, 428193.8333333333, 375949.5,
                    530340.3333333334
                ], [304249.0, 437108.0, 385692.6666666667, 487117.6666666667],
                [294026.0, 464523.0, 429530.0, 464174.0],
                [
                    266277.6666666667, 435471.3333333333, 433379.6666666667,
                    394373.6666666667
                ],
                [
                    265270.3333333333, 420530.3333333333, 459965.3333333333,
                    347322.0
                ], [306015.0, 447627.0, 540842.0, 345076.0],
                [
                    323100.3333333333, 407754.3333333333, 547333.3333333334,
                    311399.6666666667
                ],
                [
                    344572.3333333333, 364459.3333333333, 539515.3333333334,
                    291944.0
                ], [399054.0, 360568.0, 577590.0, 312511.0],
                [
                    442082.6666666667, 335357.6666666667, 575140.0,
                    331834.3333333333
                ], [493894.0, 328752.6666666667, 581087.6666666666, 377218.0],
                [502996.0, 308425.0, 539652.0, 402776.0],
                [
                    420960.3333333333, 246563.66666666666, 413487.6666666667,
                    361283.6666666667
                ],
                [
                    375614.6666666667, 226023.66666666666, 342853.0,
                    354717.3333333333
                ], [329293.0, 213215.0, 285780.0, 341374.0]]


# Waveplate [adress/id, 0 position, stepsize]
waveplates = {}
waveplates["QWP1"] = Waveplate(['beatrix', 0], 73, 0.000654)
waveplates["QWP2"] = Waveplate(['beatrix', 1], -4247, 0.000654)
waveplates["HWP1"] = Waveplate(['beatrix', 2], 777, 0.000654)

waveplates = Waveplates(waveplates)
