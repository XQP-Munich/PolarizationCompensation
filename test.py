import numpy as np
import Measurements as mm
from sympy.utilities.iterables import multiset_permutations

import Devices_Thorlabs as devices
import sys

#qwp2 = devices.K10CR1(55232924, 308.24)
#hwp = devices.K10CR1(55232454, 231.59)
#qwp1 = devices.K10CR1(55232464, 301.01)

#waveplates = devices.Waveplates([qwp1, hwp, qwp2])
source = devices.Alice_LMU()

for i in range(4):
    source.turn_on(i)

sys.exit()
# source.turn_on(1)
# waveplates.move_to(0)
# input("wait")
# waveplates.move_like_QWP(0)
# input("fin")
#b = np.fromfile("bins.bin").reshape((5, 5))
#m = np.fromfile("measure.bin", dtype=np.uint64).reshape((5, 5, 4))
#m = np.array([m[0], m[1], m[3], m[2], m[4]])
#m = np.mean(m, axis=1)

m = np.fromfile("counts.bin", dtype=np.uint64).reshape((5, 4))

# dummy data
# m = np.array([[0, 10000, 5000, 5000], [5000, 5000, 0, 10000],
#           [5000, 5000, 5000, 5000], [0, 10000, 5000, 5000],
#           [0, 0, 0, 0]])

# old data
#m = np.array(
#    [[179449.38464415, 4247.26828205, 74999.37245018, 128081.35559804],
#     [5723.12502591, 5602.73073948, 5166.80873048, 8024.14391596],
#     [0, 0, 1086.882219, 14274.18594446],
#     [0, 0, 37899.58915947, 32079.70020336],
#     [451.91630747, 400.33674648, 464.21604894, 450.72600991]])
#print(m)

coupling = np.array([0.5031, 0.4969, 0.492, 0.508])
#coupling = [0.5, 0.5, 0.5, 0.5]
ind = np.array([0, 1, 2, 3])
mt = m.copy()
ct = coupling.copy()
ps = []
results = []
for p in multiset_permutations(ind):
    print(p)
    mt[:, [0, 1, 2, 3]] = m[:, [*p]]
    ct[[0, 1, 2, 3]] = coupling[[*p]]
    res = mm.evaluate_Polcomp_Counts(mt, ct)
    ps.append(p)
    results.append(res)


for i in range(len(ps)):
    print(ps[i])
    print(results[i])
    print()
