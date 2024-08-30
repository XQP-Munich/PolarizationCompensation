import time
import numpy as np

from Devices.Templates import TIMESTAMP


class TimeTaggerUltra(TIMESTAMP):

    def __init__(self, channels):
        import TimeTagger
        print("Initializing timetagger")

        self.channel_dict = channels
        self.channels_measure = [
            channels[c]["ch"] for c in channels if c != "CLK"
        ]
        self.tt = TimeTagger.createTimeTagger()
        for key in channels:
            self.tt.setTriggerLevel(
                channels[key]["ch"] * channels[key]["edge"],
                channels[key]["trigger"])
            if key == "CLK":
                self.tt.setEventDivider(channels[key]["ch"], 1)
                self.tt.setSoftwareClock(channels[key]["ch"], 10_000_000)
        self.clock_errors = 0
        time.sleep(1)

    def read(self, t):
        import TimeTagger

        try_count = 0
        while True:
            try_count += 1
            self.stream = self.get_stream()
            self.stream.startFor(t * 1E12)
            print("Measuring the next {}s".format(t))
            self.stream.waitUntilFinished()
            print("Done")
            data = self.stream.getData()
            self.stop()
            scs = self.tt.getSoftwareClockState()
            if scs.error_counter == 0:
                self.ts = np.array([data.getChannels(), data.getTimestamps()])
                return self.ts
            print("Clock errors, trying again!")
            time.sleep(5)
            if try_count == 5:
                Exception("To many clock erros")

    def get_stream(self):
        import TimeTagger
        return TimeTagger.TimeTagStream(self.tt, 1E9, self.channels_measure)

    def get_clock_errors(self):
        scs = self.tt.getSoftwareClockState()
        self.clock_errors = scs.error_counter - self.clock_errors
        return self.clock_errors

    def stop(self):
        self.stream.stop()

    def get_counts_per_second(self):
        bins = np.arange(self.ts[1][0], self.ts[1][-1], 1E12)
        cps = []
        for i in range(0, 4):
            channel_data = self.ts[1][np.where(self.ts[0] == i + 1)]
            inds = np.digitize(channel_data, bins)
            _, counts = np.unique(inds, return_counts=True)
            cps.append(counts)
        cps = np.transpose(np.array(cps))
        return bins, cps


class TimeTaggerUltraVirtual(TimeTaggerUltra):  # Added 09.08.2024 by Peter

    def __init__(self, channels, filename="BB84_virtual"):
        import TimeTagger
        print('Virtual TimeTagger Initialized')

        self.channel_dict = channels
        self.channels_measure = [
            channels[c]["ch"] for c in channels if c != "CLK"
        ]
        # Creating Virtual TimeTagger Object
        self.tt = TimeTagger.createTimeTaggerVirtual()
        replay_sepped = -1  # < 1 is as fast as possible
        replay_begin = 0
        replay_duration = -1
        self.tt.setReplaySpeed(speed=replay_sepped)
        self.tt.replay(file=filename,
                       begin=replay_begin,
                       duration=replay_duration)

    def read(self, t=30):
        import TimeTagger

        self.stream = TimeTagger.TimeTagStream(self.tt, 1E9,
                                               self.channels_measure)
        # self.stream.waitUntilFinished()
        self.tt.waitForCompletion()
        data = self.stream.getData()
        self.ts = np.array([data.getChannels(), data.getTimestamps()])
        return self.ts
