import numpy as np
from scipy.optimize import curve_fit


def rot(theta):
    """A rotation matrix for the mueller matrix
    see https://en.wikipedia.org/wiki/Mueller_calculus
    Args:
        theta (float): The angle of rotation in degrees

    Returns:
        numpy.ndarray: The rotation matrix"""
    theta = np.deg2rad(theta)
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(2 * theta),
                      np.sin(2 * theta), 0],
                     [0, -np.sin(2 * theta),
                      np.cos(2 * theta), 0], [0, 0, 0, 1]])


def retarder(theta, delta):
    """A general retarder matrix
    see https://en.wikipedia.org/wiki/Mueller_calculus
    Args:
        theta (float): The angle of rotation in degrees
        delta (float): The retardation in radians

    Returns:
        numpy.ndarray: The retarder matrix"""
    theta = np.deg2rad(theta)
    return np.array(
        [[1, 0, 0, 0],
         [
             0,
             np.cos(2 * theta)**2 + np.sin(2 * theta)**2 * np.cos(delta),
             np.cos(2 * theta) * np.sin(2 * theta) * (1 - np.cos(delta)),
             np.sin(2 * theta) * np.sin(delta)
         ],
         [
             0,
             np.cos(2 * theta) * np.sin(2 * theta) * (1 - np.cos(delta)),
             np.cos(2 * theta)**2 * np.cos(delta) + np.sin(2 * theta)**2,
             -np.cos(2 * theta) * np.sin(delta)
         ],
         [
             0, -np.sin(2 * theta) * np.sin(delta),
             np.cos(2 * theta) * np.sin(delta),
             np.cos(delta)
         ]])


def polarizer(theta):
    """A general polarizer matrix
    see https://en.wikipedia.org/wiki/Mueller_calculus
    Args:
        theta (float): The angle of rotation in degrees

    Returns:
        numpy.ndarray: The polarizer matrix"""
    theta = np.deg2rad(theta)
    return 1 / 2 * np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0],
                             [0, 0, 0, 0]])


def printMatrix(matrix):
    """Prints a matrix niceliy
    Args:
        mat (numpy.ndarray): The matrix to print
        end (str): what to print after the matrix (default: \"\\n\")"""
    with np.printoptions(precision=3, suppress=True):
        print(matrix)


def qwp(theta):
    """A ideal lambda/4 Waveplate
    see https://en.wikipedia.org/wiki/Mueller_calculus
    Args:
        theta (float): The angle of rotation in degrees

    Returns:
        numpy.ndarray: The matrix"""
    return retarder(theta, np.pi / 2)


def hwp(theta):
    """A ideal lambda/2 Waveplate
    see https://en.wikipedia.org/wiki/Mueller_calculus
    Args:
        theta (float): The angle of rotation in degrees

    Returns:
        numpy.ndarray: The matrix"""
    return retarder(theta, np.pi)


# Math


def my_sin(x, a, omega, phi, c):

    return a * np.sin(omega * x + phi) + c


def fit_my_sin(steps, signal, p0=0):
    normalization = np.abs(np.max(signal) - np.min(signal)) / 2
    signal = signal / normalization

    if p0 == 0:
        guess_a = 1.0
        phi_step = np.mean(np.diff(steps))
        freqs = np.fft.fftfreq(len(signal), phi_step)
        spectrum = np.abs(np.fft.fft(signal)[1:len(signal) // 2 + 1])**2
        guess_omega = 2 * np.pi * freqs[np.argmax(spectrum) + 1]

        guess_phi = np.mod(-steps[np.argmax(signal)] * guess_omega + np.pi / 2,
                           2 * np.pi)
        guess_offset = np.mean(signal)
        p0 = [guess_a, guess_omega, guess_phi, guess_offset]
        print(f"p0 = {p0}")

    p, p_cov = curve_fit(my_sin, steps, signal, p0=p0)

    p[0] *= normalization
    p[2] = np.mod(p[2], 2 * np.pi)
    p[3] *= normalization
    print(f"pfit = {p}")
    return p
