import os
import Measurements as mm
import Devices_Thorlabs as devices

folder = "./data/"
if not os.path.exists(folder):
    os.mkdir(folder)

# initialize devices
source = devices.Source()
waveplates = devices.waveplates
timestamp = devices.Timestamp()

# Do this ONCE
# Caracterize Waveplates find 0 pos mark down in devices.py
#mm.

# Do this every time it is moved
# Send any Linear Pos into bob
effs = mm.get_Coupling_Efficiency_cont(folder, source, waveplates, timestamp)
#PM.turn_HWP_cont(device_adress=0,
#                 folder=folder,
#                 angle_max=180,
#                 angle_step=6,
#                 stop_at=90)

# Do this every Time
#effs = PM.eval_turn_HWP_cont(folder_effs)
#PM.POLCOMP_init_meas(folder)

#PM.POLCOMP_execute(folder, effs)
#PM.measHVPM_eval(folder, folder_effs=folder_effs)
