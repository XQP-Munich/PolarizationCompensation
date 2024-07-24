from abc import ABC, abstractmethod


class WAVEPLATE(ABC):

    @abstractmethod
    def move_to(self, pos):
        pass

    @abstractmethod
    def jog(self, direction, speed):
        pass

    @abstractmethod
    def stop(self):
        pass


class WAVEPLATES(ABC):

    def __init__(self, waveplates):
        self.waveplates = waveplates

    @abstractmethod
    def jog_like_HWP(self, speed):
        pass

    def move_to(self, pos):
        for wp in self.waveplates:
            self.waveplates[wp].move_to(pos)

    def stop(self):
        for wp in self.waveplates:
            self.waveplates[wp].stop()


class SOURCE(ABC):

    @abstractmethod
    def turn_off(self):
        pass

    @abstractmethod
    def turn_on(self, pol):
        pass


class TIMESTAMP(ABC):

    @abstractmethod
    def read(self, t):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_counts_per_second(self):
        return [], []
