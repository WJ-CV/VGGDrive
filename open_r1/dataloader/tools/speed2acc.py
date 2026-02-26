import numpy as np

def calculate_acceleration(speed, delta=0.05):
    speed = np.array(speed)
    # Configuration.
    fS = 10  # Sampling rate.
    fL = 0.2  # Cutoff frequency.
    # N = 19  # Filter length, must be odd.
    N = 19  # Filter length, must be odd.
    nn = (N + 1) // 2
    # Compute sinc filter.
    h = np.sinc(2 * fL / fS * (np.arange(N) - (N - 1) / 2))
    # Apply window.
    # h *= np.blackman(N)
    h *= np.hamming(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    h_min = h
    # h_min = minimum_phase(h, method='homomorphic')
    # h_min /= np.sum(h_min) # 再次归一化 DC 增益
    # print(h)
    # Applying the filter to a signal s can be as simple as writing
    # diff_x = ego_fut_acc_sv[1, 0] - ego_fut_acc_sv[0, 0]
    # diff_y = ego_fut_acc_sv[1, 1] - ego_fut_acc_sv[0, 1]
    ###############################################################
    filtered_speed = np.convolve(speed[:], h_min, mode="same")
    # fL = 2  # Cutoff frequency.
    # h *= np.blackman(N)
    # h /= np.sum(h)
    # h_min_2 = h
    # # h_min_2 = minimum_phase(h, method='homomorphic')
    # # h_min_2 /= np.sum(h_min_2)         # 再次归一化 DC 增益
    # ego_fut_vel_filter_x = np.convolve(ego_fut_vel_filter_x, h_min_2, mode = 'same')
    # ego_fut_vel_filter_y = np.convolve(ego_fut_vel_filter_y, h_min_2, mode = 'same')
    filtered_acc = np.diff(speed, axis=0) / delta
    return filtered_acc.tolist()
