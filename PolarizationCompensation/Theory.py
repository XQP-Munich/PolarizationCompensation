import numpy as np
import itertools
import scipy.optimize as so


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

    p, p_cov = so.curve_fit(my_sin, steps, signal, p0=p0)

    p[0] *= normalization
    p[2] = np.mod(p[2], 2 * np.pi)
    p[3] *= normalization
    return p


def produce_density_matrix(vec):
    """
    Input: density matrix from Jones Input-Ket
    """
    rho = np.outer(vec, np.conjugate(vec))

    return rho


def detection_statistics_HVPMRL(U, Jones_in):
    rho = produce_density_matrix(Jones_in)
    rho_twist = U @ rho @ np.linalg.inv(U)

    P = np.array([
        np.abs(rho_twist[0][0]),
        np.abs(rho_twist[1][1]),
        1 / 2 * np.abs(rho_twist[0][0] + rho_twist[1][1] + rho_twist[0][1] +
                       rho_twist[1][0]),
        1 / 2 * np.abs(rho_twist[0][0] + rho_twist[1][1] - rho_twist[0][1] -
                       rho_twist[1][0]),
        1 / 2 * np.abs(rho_twist[0][0] - 1j * rho_twist[0][1] +
                       1j * rho_twist[1][0] + rho_twist[1][1]),
        1 / 2 * np.abs(rho_twist[0][0] + 1j * rho_twist[0][1] -
                       1j * rho_twist[1][0] + rho_twist[1][1]),
    ])

    return P


def lin_ret(θ, η):
    """
    Returns Jones matrix for general linear retarder with phase shift eta and at angle theta with respect to lab-frame
    """
    c_t = np.cos(θ)
    s_t = np.sin(θ)
    exp_i_e = np.exp(1j * η)

    J = np.exp(-1j * η / 2) * np.array([[
        c_t**2 + exp_i_e * s_t**2, (1 - exp_i_e) * c_t * s_t
    ], [(1 - exp_i_e) * c_t * s_t, s_t**2 + exp_i_e * c_t**2]])

    return J


def fit_Q_Channel(measurement, waveplate_setup):
    retarder_map = {"HWP": np.pi, "QWP": np.pi / 2}
    q_Channel = lambda x: lin_ret(x[2], retarder_map[waveplate_setup[
        2]]) @ lin_ret(x[1], retarder_map[waveplate_setup[1]]) @ lin_ret(
            x[0], retarder_map[waveplate_setup[0]])
    H = (1, 0)
    V = (0, 1)
    P = (1 / np.sqrt(2), 1 / np.sqrt(2))
    M = (1 / np.sqrt(2), -1 / np.sqrt(2))
    input_States = [H, V, P, M]

    def objective(x):
        error = 0
        count = 0
        for state in input_States:
            error += sum(
                np.abs(
                    detection_statistics_HVPMRL(np.linalg.inv(q_Channel(
                        x)), state) - measurement[count])**2)
            count += 1
        return error

    bounds = [(0, np.pi), (0, np.pi), (0, np.pi)]

    measurement = np.array(measurement)
    alpha_s = np.linspace(0, np.pi, 5)
    beta_s = np.linspace(0, np.pi, 5)
    gamma_s = np.linspace(0, np.pi, 5)
    Roots = []
    funs = []
    for x0 in itertools.product(alpha_s, beta_s, gamma_s):
        x0 = np.array(x0)
        Root = so.minimize(objective, x0=x0, bounds=bounds)
        Roots.append(Root.x)
        funs.append(Root.fun)

    Roots = np.array(Roots)
    min_fun_ind = np.where(funs == min(funs))[0][0]

    x = Roots[min_fun_ind]
    fun = funs[min_fun_ind]

    angles = np.rad2deg(x)

    if funs[min_fun_ind] > .1:
        print("Optimization failed.")

    return angles, fun


def angle_between_stokes_vectors(v1, v2):
    """
    Calculate the angle in radians between two 3D vectors.

    Parameters:
    - v1, v2: NumPy arrays representing the two 3D vectors.

    Returns:
    - angle: The angle in radians between the two vectors.
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle = angle / (2 * np.pi) * 360
    return angle
