from abc import ABC, abstractmethod


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
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_counts_per_second(self):
        return [], []
