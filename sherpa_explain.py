import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
import pandas as pd


def load_shapvalues(name):
    """
    Load SHAP values
    :param name: Name of directory where SHAP values are stored
    :return: SHAP values
    """
    shapname = "cele_res/cnn_" + name + "/shap_cnn.npy"
    shap_values = np.load(shapname)
    print("Shape SHAP values: ", shap_values.shape)
    return shap_values


def multi_lineplot(face_sums, blur_sums, scram_sums, name):
    """
    Draw line plot for all time points of the SHAP values
    :param face_sums: SHAP values for target class face
    :param blur_sums: SHAP values for target class blur
    :param scram_sums: SHAP values for target class scram
    :param name: Name of the classification task
    """
    def to_secs(x):
        return x * (1500 / 768)

    fig, ax = plt.subplots()
    face_line, = ax.plot(face_sums, label="face", color="r", alpha=1)
    blur_line, = ax.plot(blur_sums, label="blurred", color="orange", alpha=0.75)
    scram_line, = ax.plot(scram_sums, label="scrambled", color="b", alpha=0.6)
    ax.legend(handles=[face_line, blur_line, scram_line])
    ax.set_xlabel("Data points")
    ax.set_ylabel("Sum of SHAP values")
    secax = ax.secondary_xaxis('top', functions=(to_secs, to_secs))
    secax.set_xlabel('Milliseconds')

    p = "cele_res/cnn_" + name + "/line_plot_targets" + ".png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.show()


def find_extrema(shap_timesums=np.ndarray, order=20):
    """
    Find local extrema in 1D array
    :param shap_timesums: 1D vector mean over electrodes; contains one shap value per time point
    :param order: Number of time points around extremum that have to be less extreme in order to make it count
    :return: Array with the time indices of local maximums and local minimums respectively
    """
    maxs = argrelmax(shap_timesums, order=order)
    mins = argrelmin(shap_timesums, order=order)
    return maxs, mins


def find_electrodes(name, targetname, shap_vals, timepoint, windowsize=40):
    """
    Find the most important electrodes for given time window
    :param name: Name of directory with SHAP values
    :param targetname: Name of the target class
    :param shap_vals: 2D SHAP values, all time points x all electrodes
    :param timepoint: The chosen local maximum from the "find-extrema" function
    :param windowsize: Total size of the window around the local extremum in ms
    :return: Dataframe with all electrodes, their SHAP values and quantiles; sum of all shap values
    """

    win_in_dt = windowsize / (1500 / 768)  # windowsize (ms) to datapoints
    # formula works only for 1.5 sec epochs and 512 hz sampling rate
    d = int(win_in_dt / 2)
    r = range(timepoint-d, timepoint+d)

    window = shap_vals[r, :]
    peak = shap_vals[timepoint, :]
    shap_sum = np.sum(peak)
    # mean over time window
    mean_window = np.mean(window, axis=0)
    # put into dict in order to have the electrode indices
    edict = dict(zip(range(1, 129), mean_window))  # 128 electrodes

    # add all index/shap value pairs to list that are higher than value given by quantile
    quantils = []
    q = [0.25, 0.5, 0.75, 0.9]
    q_vals = np.quantile(list(edict.values()), q)
    for k,v in edict.items():
        if v >= q_vals[3]:
            quantils.append((k, v, q[3]))
        elif v >= q_vals[2] and v < q_vals[3]:
            quantils.append((k, v, q[2]))
        elif v >= q_vals[1] and v < q_vals[2]:
            quantils.append((k, v, q[1]))
        elif v >= q_vals[0] and v < q_vals[1]:
            quantils.append((k, v, q[0]))
        else:
            quantils.append((k, v, 0))
    # sort result list according to shap value (descending)
    eq_sorted = sorted(quantils, key=lambda tup: tup[1], reverse=True)
    eq_df = pd.DataFrame(eq_sorted, columns=['electrode', 'shap_value', 'quantile'])
    # save to csv
    p = "cele_res/cnn_" + name + "/electrode_df_" + targetname + ".csv"
    eq_df.to_csv(p, index=False)

    return eq_df, shap_sum


def find_coordinates(shap_vals, shap_sums, order, windowsize, targetname):
    """
    Find temporal and spatial coordinates for EEG component
    :param shap_vals: SHAP values
    :param shap_sums: SHAP values summed up for one target class
    :param order: Number of time points around extremum that have to be less extreme in order to make it count,
                    defaults to 20
    :param windowsize: Total size of the window around the local extremum in ms, defaults to 40
    :param targetname: Name of the target class analyzed
    :return:
    """
    print(targetname)
    maxs, mins = find_extrema(shap_sums, order)
    # local extremum has to be max because we took the absolute values of the shap values
    max_values = dict(zip(maxs[0].tolist(), shap_sums[maxs].tolist()))
    if targetname == "face_peak2":
        max_values_sorted = sorted(max_values, key=max_values.get, reverse=True)
        locex = max_values_sorted[1]
        locex_ms = locex * (1500 / 768)
    else:
        locex = max(max_values, key=max_values.get)
        locex_ms = locex * (1500 / 768)
    print(f"Find most important electrodes for time window around time point {locex}, ms {locex_ms}: ")
    eq_df, shap_sum = find_electrodes(name, targetname, shap_vals, locex, windowsize)
    print(eq_df.head(10))
    print(f"All electrodes have a summed shap value of %.4f.\n" % shap_sum)
    return eq_df


def windowed_shap(name, order, windowsize):
    """
    Wrapper function to find EEG component
    :param name: Name of the classification task
    :param order: Number of time points around extremum that have to be less extreme in order to make it count,
                    defaults to 20
    :param windowsize: Total size of the window around the local extremum in ms, defaults to 40
    """
    shap_values = load_shapvalues(name)
    # mean over trials and then sum electrodes for every target class separately
    face_shap = np.mean(np.abs(shap_values[0]), axis=0)
    face_sums = np.sum(face_shap, axis=1)
    blur_shap = np.mean(np.abs(shap_values[1]), axis=0)
    blur_sums = np.sum(blur_shap, axis=1)
    scram_shap = np.mean(np.abs(shap_values[2]), axis=0)
    scram_sums = np.sum(scram_shap, axis=1)

    mean_shap = np.mean(np.abs(shap_values), axis=(0, 1))
    mean_sums = np.sum(mean_shap, axis=1)

    print("Start looking for a window...")
    multi_lineplot(face_sums, blur_sums, scram_sums, name)
    eq_face = find_coordinates(face_shap, face_sums, order, windowsize, "face")
    eq_face_peak2 = find_coordinates(face_shap, face_sums, order, windowsize, "face_peak2")
    eq_blur = find_coordinates(blur_shap, blur_sums, order, windowsize, "blurred")
    eq_scram = find_coordinates(scram_shap, scram_sums, order, windowsize, "scrambled")
    eq_mean = find_coordinates(mean_shap, mean_sums, order, windowsize, "mean")


if __name__ == '__main__':
    name = 'small_pc'
    windowed_shap(name=name, order=20, windowsize=40)
