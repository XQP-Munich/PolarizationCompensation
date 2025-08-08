from abc import ABC, abstractmethod
import numpy as np


class WAVEPLATE(ABC):
    @abstractmethod
    def move_to(self, pos):
        pass

    @abstractmethod
    def wait_move(self):
        pass

    @abstractmethod
    def setup_jog(self, speed):
        pass

    @abstractmethod
    def jog(self, direction):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def home(self):
        pass

    @abstractmethod
    def wait_home(self):
        pass


class WAVEPLATES(ABC):
    @abstractmethod
    def jog_like_HWP(self, speed):
        raise Exception("<WAVEPLATES> jog_like_HWP not implemented")

    @abstractmethod
    def move_like_HWP(self, angle):
        raise Exception("<WAVEPLATES> move_like_HWP not implemented")

    @abstractmethod
    def move_like_QWP(self, angle):
        raise Exception("<WAVEPLATES> move_like_QWP not implemented")

    @abstractmethod
    def stop(self):
        raise Exception("<WAVEPLATES> stop not implemented")

    @abstractmethod
    def home(self):
        raise Exception("<WAVEPLATES> home not implemented")

    @abstractmethod
    def move_to(self, angle):
        raise Exception("<WAVEPLATES> move_to not implemented")


class SOURCE(ABC):
    @abstractmethod
    def turn_off(self):
        pass

    @abstractmethod
    def turn_on(self, pol):
        pass

    @abstractmethod
    def send_key(self, key):
        pass


class TIMESTAMP(ABC):
    @abstractmethod
    def read(self, t):
        raise Exception("<TIMESTAMP> read not yet implemented")

    @abstractmethod
    def stop(self):
        raise Exception("<TIMESTAMP> stop not yet implemented")

    @abstractmethod
    def get_counts_per_second(self, t) -> tuple[np.ndarray, np.ndarray]:
        raise Exception("<TIMESTAMP> get_counts_per_second not yet implemented")
