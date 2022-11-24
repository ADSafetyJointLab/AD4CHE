from cutin_extraction import ScenarioExtraction
import utils
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np


def plot_kde(data, bins, ax):
    arr = np.array(data)
    X = np.linspace(bins.min(), bins.max(), 1000)[:, np.newaxis]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=2)
    kde_model.fit(arr[:, np.newaxis])
    log_dens = kde_model.score_samples(X)
    ax.plot(
        X[:, 0],
        np.exp(log_dens),
        color="#FF8D57",
        lw=2,
        linestyle="-",
        label="kernel = '{0}'".format('gaussian'),
    )


# plot initial relative longitudinal distance
def plot_dx0(data):
    fig, ax = plt.subplots(1, 1)
    num = 5
    bins_overlap = np.linspace(min(data), 0, num)
    bins_nonoverlap = np.linspace(0, max(data), num)
    bins = np.concatenate((bins_overlap[:-1], bins_nonoverlap))

    n, bins, patches = ax.hist(data, bins, density=True, histtype='bar', rwidth=1.0, edgecolor='black',
                               label='Overlap cut-in')
    # set facecolor for bars
    for i in range(len(bins) - 1):
        if bins[i] < 0:
            patches[i].set_facecolor("#D49AB3")
        else:
            patches[i].set_facecolor("#7A93B2")

    plot_kde(data, bins, ax)
    ax.set_xlabel(r'Relative longitudinal distance d$_{x,0}$ [m]')
    ax.set_ylabel('Density')
    ax.grid(axis='y', alpha=0.75)
    # red_patch = mpatches.Patch(color='#D49AB3', label='Overlap cut-in')
    blue_patch = mpatches.Patch(color='#7A93B2', label='Non-overlap cut-in')
    # line1 = mpatches.Patch(color='#FF8D57', label='KDE')
    leg = ax.legend(loc="upper left")
    handles = leg.legendHandles
    handles.append(blue_patch)
    ax.legend(handles=handles)


# plot initial velocity of the ego vehicle
def plot_ve0(overlap_data, nonoverlap_data):
    fig, ax = plt.subplots(2)
    num = 6
    bins_overlap = np.linspace(min(overlap_data), max(overlap_data), num)
    bins_nonoverlap = np.concatenate((bins_overlap[:-1], np.linspace(max(overlap_data), max(nonoverlap_data), 2)))
    n, bins, patches = ax[0].hist(overlap_data, bins_overlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#D49AB3', edgecolor='black')
    plot_kde(overlap_data, bins, ax[0])

    n, bins, patches = ax[1].hist(nonoverlap_data, bins_nonoverlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#7A93B2', edgecolor='black')
    plot_kde(nonoverlap_data, bins, ax[1])

    ax[0].set_xlim([0, 25])
    ax[1].set_xlim([0, 25])
    ax[1].set_ylim([0, 0.17])
    ax[1].set_xlabel(r'Initial ego velocity v$_{\rm e,0}$ [m/s]')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    ax[0].grid(alpha=0.75)
    ax[1].grid(alpha=0.75)
    ax[0].legend(['Overlap cut-in'])
    ax[1].legend(['Non-overlap cut-in'])


def main():
    obj = utils.load_object("data.pickle")

    overlap_dx0 = []
    nonoverlap_dx0 = []
    overlap_ve0 = []
    nonoverlap_ve0 = []
    # data from overlap
    for key, car_pair in obj.overlap_car_pairs.items():
        cutin_car = car_pair[0]
        ego = car_pair[1]

        rel_d_x = None
        # initial width of each vehicle at first frame
        width = float(cutin_car[0][obj.width]) / 2 + float(ego[0][obj.width]) / 2
        # initial relative distance dx0
        if utils.driving_direction(obj.vx, cutin_car) == 1:
            rel_d_x = float(cutin_car[0][obj.x]) - float(ego[0][obj.x]) - width

        elif utils.driving_direction(obj.vx, cutin_car) == -1:
            rel_d_x = float(ego[0][obj.x]) - float(cutin_car[0][obj.x]) - width

        if rel_d_x is not None:
            overlap_dx0.append(rel_d_x)
        else:
            raise 'dx0 is not found'

        # initial ego velocity
        overlap_ve0.append(abs(float(ego[0][obj.vx])))

    # data from non-overlap
    for key, car_pair in obj.nonoverlap_car_pairs.items():
        cutin_car = car_pair[0]
        ego = car_pair[1]

        rel_d_x = None
        # initial width of each vehicle at first frame
        width = float(cutin_car[0][obj.width]) / 2 + float(ego[0][obj.width]) / 2
        # initial relative distance dx0
        if utils.driving_direction(obj.vx, cutin_car) == 1:
            rel_d_x = float(cutin_car[0][obj.x]) - float(ego[0][obj.x]) - width

        elif utils.driving_direction(obj.vx, cutin_car) == -1:
            rel_d_x = float(ego[0][obj.x]) - float(cutin_car[0][obj.x]) - width

        if rel_d_x is not None:
            nonoverlap_dx0.append(rel_d_x)
        else:
            raise 'dx0 is not found'

        # initial ego velocity
        nonoverlap_ve0.append(abs(float(ego[0][obj.vx])))

    plot_dx0(overlap_dx0 + nonoverlap_dx0)
    plot_ve0(overlap_ve0, nonoverlap_ve0)

    plt.show()


if __name__ == '__main__':
    main()
