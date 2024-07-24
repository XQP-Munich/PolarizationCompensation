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

    p, p_cov = so.curve_fit(my_sin, steps, signal, p0=p0)

    p[0] *= normalization
    p[2] = np.mod(p[2], 2 * np.pi)
    p[3] *= normalization
    print(f"pfit = {p}")
    return p


H = (1, 0)
V = (0, 1)
P = (1 / np.sqrt(2), 1 / np.sqrt(2))
M = (1 / np.sqrt(2), -1 / np.sqrt(2))
Set_In = [H, V, P, M]


def produce_density_matrix(vec):
    """
    Input: density matrix from Jones Input-Ket
    """
    rho = np.outer(vec, np.conjugate(vec))

    return rho


def detection_statistics(U, Jones_in):
    """
    Given unitary transformation U, and Jones input state, detections statistics for measuring H, V, P and M will
    be given in this order. 
    """
    rho = produce_density_matrix(Jones_in)

    rho_twist = np.matmul(np.matmul(U, rho), np.linalg.inv(U))

    P = [
        np.abs(rho_twist[0][0]),
        np.abs(rho_twist[1][1]),
        1 / 2 * np.abs(rho_twist[0][0] + rho_twist[1][1] + rho_twist[0][1] +
                       rho_twist[1][0]),
        1 / 2 * np.abs(rho_twist[0][0] + rho_twist[1][1] - rho_twist[0][1] -
                       rho_twist[1][0])
    ]

    return P


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
    s_e = np.sin(η)
    exp_i_e = np.exp(1j * η)

    J = np.exp(-1j * η / 2) * np.array([[
        c_t**2 + exp_i_e * s_t**2, (1 - exp_i_e) * c_t * s_t
    ], [(1 - exp_i_e) * c_t * s_t, s_t**2 + exp_i_e * c_t**2]])

    return J


def Q_Channel(x):
    """
    Returns unitary transformation caused by two quater and one half-wave-plate. 
    """

    α, β, γ = x
    J_Q = np.matmul(np.matmul(lin_ret(γ, np.pi), lin_ret(β, np.pi / 2)),
                    lin_ret(α, np.pi / 2))

    #J_Q = np.matmul(np.matmul(lin_ret(γ, np.pi / 2), lin_ret(β, np.pi)),
    #                lin_ret(α, np.pi / 2))

    return J_Q


def fit_Q_Channel(measurement):

    def objective(x):
        error = 0
        count = 0
        for state in Set_In:
            error += sum(
                np.abs(
                    detection_statistics_HVPMRL(np.linalg.inv(Q_Channel(
                        x)), state) - measurement[count])**2)
            count += 1

        return error

    bounds = [(0, np.pi), (0, np.pi), (0, np.pi / 2)]
    #bounds = [(0, np.pi), (0, np.pi / 2), (0, np.pi)]
    #bounds = [(0, 0, 0), (np.pi, np.pi, np.pi/2)]

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
    print('best angles set: ', x)
    print('best loss: ', funs[min_fun_ind])

    ##### Virtual Quantum Channel to see if Output fits Measurement
    sim_measurement = []
    for state in Set_In:
        sim_measurement.append(detection_statistics(Q_Channel(x), state))

    sim_measurement = np.array(sim_measurement)

    #print('Measurement after Quantum Channel\n', measurement)
    #print('Measurement after fit Quantum Channel\n', sim_measurement)

    angles = np.rad2deg(x)

    if funs[min_fun_ind] > .005:
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
