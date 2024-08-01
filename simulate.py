import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute

from PolarizationCompensation.Theory import qwp, hwp, printMatrix


# Change this to waveplate setup
def model(alpha, beta, gamma):
    return qwp(gamma) @ hwp(beta) @ qwp(alpha)


def get_angles(matrix, x0=[0, 0, 0], output=False, r=[8, 8, 8], Ns=10):

    def f(val):
        alpha, beta, gamma = val
        A = model(alpha, beta, gamma) - matrix
        return abs(A).sum()

    r = np.array(r)
    x0 = np.array(x0)

    a, b, g = x0
    ar, br, gr, = x0 + r
    res = brute(f, ((a, ar), (b, br), (g, gr)), Ns=Ns, full_output=True)
    alpha, beta, gamma = [x % 360 for x in res[0]]
    if output:
        print(
            "alpha = {:.1f}, beta = {:.1f}, gamma = {:.1f} with error = {:.1f}"
            .format(alpha, beta, gamma, res[1]))
        print()
        print("target matrix:")
        printMatrix(matrix)
        print()

        print("fit matrix:")
        printMatrix(model(alpha, beta, gamma))
        print()

    return alpha % 360, beta % 360, gamma % 360, res[1]


positions = [[0, 0, 90]]
xs = []
es = []
first = True
for i in np.arange(0, 181, 10):
    r = [8, 8, 8]
    if first:
        r = [0, 0, 0]
        first = False
    a, b, g, e = get_angles(hwp(i), output=False, x0=positions[-1], r=r)
    positions.append([i, b, i + 90])
    xs.append(i)
    es.append(e)
    if e > 2:
        print("Error too high")
        break

positions = np.array(positions[1:])
pt = np.transpose(positions)

a, b, g, e = get_angles(qwp(0), output=False, x0=[0, 0, 0], r=[180, 180, 180])
print("Values for fixed QWP at 0deg: alpha={:.2f}, beta={:.2f}, gamma={:.2f}".
      format(a, b, g))

for i, ys in enumerate(pt):
    plt.plot(xs, ys, label=i)
plt.plot(xs, np.array(es) * 10**7, label="error *10^7")
plt.xlabel("Rotation angle of simulated HWP in deg")
plt.ylabel("Rotation angle of actual WPs in deg")
plt.legend()
plt.show()
