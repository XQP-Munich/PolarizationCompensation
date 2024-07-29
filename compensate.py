import os
import PolarizationCompensation.Measurements as mm
import PolarizationCompensation.Devices_Thorlabs as devices

folder = "./data/"
if not os.path.exists(folder):
    os.mkdir(folder)

# initialize devices
qwp2 = devices.K10CR1(55232924, 308.24)
hwp = devices.K10CR1(55232454, 231.59)
qwp1 = devices.K10CR1(55232464, 301.01)

waveplates = devices.Waveplates([qwp1, hwp, qwp2])
source = devices.Alice_LMU()
timestamp = devices.TimeTaggerUltra()

# Do this ONCE
# Caracterize Waveplates find 0 pos mark down in devices.py
#mm.

# Do this every time it is moved
# Send any Linear Pos into bob
coupling = mm.get_Coupling_Efficiency_cont(source, waveplates, timestamp)
print(coupling)
#PM.turn_HWP_cont(device_adress=0,
#                 folder=folder,
#                 angle_max=180,
#                 angle_step=6,
#                 stop_at=90)

# Do this every Time
bins, counts = mm.measure_Polcomp_Counts(source, waveplates, timestamp)
counts.tofile("counts.bin")
print(counts)
angles, loss, stokesangles = mm.evaluate_Polcomp_Counts(counts, coupling)
print(angles, loss, stokesangles)
waveplates.move_to(angles)
