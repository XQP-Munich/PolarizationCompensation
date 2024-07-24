from theory import qwp, hwp, printMatrix
import numpy as np
from scipy.optimize import brute

import matplotlib.pyplot as plt
#for i in np.arange(0, 180, 22.5):
#    printMatrix(qwp(i) @ qwp(i + 90))
#    print(qwp(i))


def get_angles(matrix, x0=[0, 0, 0], output=False, first=False):

    def model(alpha, beta, gamma):
        return qwp(gamma) @ hwp(beta) @ qwp(alpha)

    def f(val):
        alpha, beta, gamma = val
        A = model(alpha, beta, gamma) - matrix
        return abs(A).sum()

    a, b, g = x0
    r = 8
    if first:
        r = 180
    res = brute(f, ((a,a+r),(b, b + r), (g, g + r)), Ns=100, full_output=True)
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
for i in np.arange(0, 181, 180):
    print(i)
    print(positions[-1])
    a, b, g, e = get_angles(qwp(0), output=True, x0=positions[-1], first=first)
    positions.append([a, b, g])
    xs.append(i)
    es.append(e)
    first = False

positions = np.array(positions[1:])
print(positions)
pt = np.transpose(positions)

print(pt)
for i, ys in enumerate(pt):
    plt.plot(xs, ys, label=i)
plt.plot(xs, np.array(es) * 10**7, label="error *10^7")
plt.legend()
plt.show()
