from cutin_extraction import ScenarioExtraction
import utils
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def plot_kde(data, bins, ax, bandwidth):
    arr = np.array(data)
    X = np.linspace(bins.min(), bins.max(), 1000)[:, np.newaxis]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
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
def plot_rel_dx0(data):
    fig, ax = plt.subplots(1, 1)
    bins_overlap = np.linspace(min(data), 0, 3)
    bins_nonoverlap = np.linspace(0, max(data), 10)
    bins = np.concatenate((bins_overlap[:-1], bins_nonoverlap))

    n, bins, patches = ax.hist(data, bins, density=True, histtype='bar', rwidth=1.0, edgecolor='black',
                               label='Overlap cut-in')
    # set facecolor for bars
    for i in range(len(bins) - 1):
        if bins[i] < 0:
            patches[i].set_facecolor("#D49AB3")
        else:
            patches[i].set_facecolor("#7A93B2")

    plot_kde(data, bins, ax, 2)
    ax.set_xlabel(r'Relative longitudinal distance d$_{\rm rel,x0}$ [m]')
    ax.set_ylabel('Density')
    ax.grid(axis='y', alpha=0.75)
    # red_patch = mpatches.Patch(color='#D49AB3', label='Overlap cut-in')
    blue_patch = mpatches.Patch(color='#7A93B2', label='Non-overlap cut-in')
    leg = ax.legend(loc="upper left")
    handles = leg.legendHandles
    handles.append(blue_patch)
    ax.legend(handles=handles)


def plot_rel_vx0(overlap_data, nonoverlap_data):
    fig, ax = plt.subplots(2)
    num = 6
    bins_overlap = num
    bins_nonoverlap = 2*(num -1)
    n, bins, patches = ax[0].hist(overlap_data, bins_overlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#D49AB3', edgecolor='black')
    plot_kde(overlap_data, bins, ax[0], 1)

    n, bins, patches = ax[1].hist(nonoverlap_data, bins_nonoverlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#7A93B2', edgecolor='black')
    plot_kde(nonoverlap_data, bins, ax[1], 1)

    ax[0].set_xlim([-7, 5.6])
    ax[1].set_xlim([-7, 5.6])
    ax[0].set_ylim([0, 0.29])
    ax[1].set_ylim([0, 0.29])
    ax[1].set_xlabel(r'Initial relative longitudinal velocity v$_{\rm rel,x0}$ [m/s]')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    ax[0].grid(alpha=0.75)
    ax[1].grid(alpha=0.75)
    ax[0].legend(['Overlap cut-in'])
    ax[1].legend(['Non-overlap cut-in'])


# plot initial velocity of the ego vehicle
def plot_ve0(overlap_data, nonoverlap_data):
    fig, ax = plt.subplots(2)
    num = 6
    bins_overlap = np.linspace(min(overlap_data), max(overlap_data), 2*(num-1))
    bins_nonoverlap = np.concatenate((bins_overlap[:-1], np.linspace(max(overlap_data), max(nonoverlap_data), 3)))
    n, bins, patches = ax[0].hist(overlap_data, bins_overlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#D49AB3', edgecolor='black')
    plot_kde(overlap_data, bins, ax[0], 2)

    n, bins, patches = ax[1].hist(nonoverlap_data, bins_nonoverlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#7A93B2', edgecolor='black')
    plot_kde(nonoverlap_data, bins, ax[1], 2)

    ax[0].set_xlim([0, 25])
    ax[1].set_xlim([0, 25])
    ax[1].set_ylim([0, 0.17])
    ax[1].set_xlabel(r'Initial ego velocity v$_{\rm ego,0}$ [m/s]')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    ax[0].grid(alpha=0.75)
    ax[1].grid(alpha=0.75)
    ax[0].legend(['Overlap cut-in'])
    ax[1].legend(['Non-overlap cut-in'])


# plot maximum lateral velocity of the challenging vehicle
def plot_max_vy(overlap_data, nonoverlap_data):
    fig, ax = plt.subplots(2)
    num = 6
    bins_overlap = num
    bins_nonoverlap = 2*(num-1)
    n, bins, patches = ax[0].hist(overlap_data, bins_overlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#D49AB3', edgecolor='black')
    plot_kde(overlap_data, bins, ax[0], 0.1)

    n, bins, patches = ax[1].hist(nonoverlap_data, bins_nonoverlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#7A93B2', edgecolor='black')
    plot_kde(nonoverlap_data, bins, ax[1], 0.1)

    ax[0].set_xlim([0, 1.5])
    ax[1].set_xlim([0, 1.5])
    ax[0].set_ylim([0, 2.5])
    ax[1].set_ylim([0, 2.5])
    ax[1].set_xlabel(r'Maximum lateral velocity of the challenging vehicle v$_{\rm cha,y,\rm max}$ [m/s]')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    ax[0].grid(alpha=0.75)
    ax[1].grid(alpha=0.75)
    ax[0].legend(['Overlap cut-in'])
    ax[1].legend(['Non-overlap cut-in'])


def plot_max_offset(overlap_data, nonoverlap_data):
    fig, ax = plt.subplots(2)
    num = 6
    bins_overlap = 2*(num - 1)
    bins_nonoverlap = 2*(num + 1)
    n, bins, patches = ax[0].hist(overlap_data, bins_overlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#D49AB3', edgecolor='black')
    plot_kde(overlap_data, bins, ax[0], 0.1)

    n, bins, patches = ax[1].hist(nonoverlap_data, bins_nonoverlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#7A93B2', edgecolor='black')
    plot_kde(nonoverlap_data, bins, ax[1], 0.1)

    ax[0].set_xlim([0, 2])
    ax[1].set_xlim([0, 2])
    # ax[0].set_ylim([0, 2.5])
    # ax[1].set_ylim([0, 2.5])
    ax[1].set_xlabel(r'Ego maximum lateral offset [m] during [$\it T$$_{\rm 1}$, $\it T$$_{\rm 3}$]')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    ax[0].grid(alpha=0.75)
    ax[1].grid(alpha=0.75)
    ax[0].legend(['Overlap cut-in'])
    ax[1].legend(['Non-overlap cut-in'])


def plot_duration(overlap_data, nonoverlap_data):
    fig, ax = plt.subplots(2)
    num = 6
    bins_overlap = num
    bins_nonoverlap = num
    n, bins, patches = ax[0].hist(overlap_data, bins_overlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#D49AB3', edgecolor='black')
    plot_kde(overlap_data, bins, ax[0], 2)

    n, bins, patches = ax[1].hist(nonoverlap_data, bins_nonoverlap, density=True, histtype='bar', rwidth=1.0,
                                  facecolor='#7A93B2', edgecolor='black')
    plot_kde(nonoverlap_data, bins, ax[1], 2)

    # ax[0].set_xlim([0, 2.5])
    # ax[1].set_xlim([0, 2.5])
    # ax[0].set_ylim([0, 2.5])
    # ax[1].set_ylim([0, 2.5])
    ax[1].set_xlabel('Duration of cut-in maneuvers [s]')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    ax[0].grid(alpha=0.75)
    ax[1].grid(alpha=0.75)
    ax[0].legend(['Overlap cut-in'])
    ax[1].legend(['Non-overlap cut-in'])


class ParameterDistribution:
    def __init__(self,
                 frame: int,
                 width: int,
                 vx: int,
                 vy: int,
                 x: int,
                 ego_offset: int,
                 ):

        self.frame = frame
        self.width = width
        self.vx = vx
        self.vy = vy
        self.x = x
        self.ego_offset = ego_offset

    def get_data(self, car_pairs):
        rel_dx0 = []
        rel_vx0 = []
        ve0 = []
        vy_max = []
        offset_max = []
        duration = []

        for key, car_pair in car_pairs.items():
            cutin_car = car_pair[0]
            ego = car_pair[1]
            key_timestamps_t1 = int(car_pair[2][0])
            key_timestamps_t3 = int(car_pair[2][1])
            key_timestamps_t5 = int(car_pair[2][2])
            # 30 is the fps
            duration.append((int(key_timestamps_t5) - int(key_timestamps_t1)) / 30)

            rel_d_x = None
            rel_v_x = None
            # initial size of each vehicle at first frame
            width = float(cutin_car[0][self.width]) / 2 + float(ego[0][self.width]) / 2
            # initial relative distance dx0, relative longitudinal velocity dvx0
            if utils.driving_direction(self.vx, cutin_car) == 1:
                rel_d_x = float(cutin_car[0][self.x]) - float(ego[0][self.x]) - width
                rel_v_x = float(cutin_car[0][self.vx]) - float(ego[0][self.vx])
            elif utils.driving_direction(self.vx, cutin_car) == -1:
                rel_d_x = float(ego[0][self.x]) - float(cutin_car[0][self.x]) - width
                rel_v_x = float(ego[0][self.vx]) - float(cutin_car[0][self.vx])

            # initial relative longitudinal distance rel_dx0
            rel_dx0.append(rel_d_x)
            # initial ego velocity
            ve0.append(abs(float(ego[0][self.vx])))
            # relative longitudinal velocity rel_vx0
            rel_vx0.append(rel_v_x)
            # maximum lateral velocity of the cut-in vehicle
            vy = [abs(float(car[self.vy])) for car in cutin_car]
            vy_max.append(max(vy))
            # Lateral offset of the ego vehicle at t3
            y_offset = []
            for car in cutin_car:
                if key_timestamps_t1 <= int(car[self.frame]) <= key_timestamps_t3:
                    y_offset.append(abs(float(car[self.ego_offset])))
                elif int(car[self.frame]) > key_timestamps_t3:
                    break
            offset_max.append(max(y_offset))

        return rel_dx0, rel_vx0, ve0, vy_max, offset_max, duration


def main():
    obj = utils.load_object("data.pickle")
    para_data = ParameterDistribution(obj.frame, obj.width, obj.vx, obj.vy, obj.x, obj.ego_offset)
    overlap_rel_dx0, overlap_rel_vx0, overlap_ve0, overlap_vy_max, overlap_offset_max, overlap_duration = para_data.get_data(
        obj.overlap_car_pairs)
    nonoverlap_rel_dx0, nonoverlap_rel_vx0, nonoverlap_ve0, nonoverlap_vy_max, nonoverlap_offset_max, nonoverlap_duration = para_data.get_data(
        obj.nonoverlap_car_pairs)

    # plot the results
    plot_rel_dx0(overlap_rel_dx0 + nonoverlap_rel_dx0)
    plot_rel_vx0(overlap_rel_vx0, nonoverlap_rel_vx0)
    plot_ve0(overlap_ve0, nonoverlap_ve0)
    plot_max_vy(overlap_vy_max, nonoverlap_vy_max)
    plot_max_offset(overlap_offset_max, nonoverlap_offset_max)
    plot_duration(overlap_duration, nonoverlap_duration)

    plt.show()


if __name__ == '__main__':
    main()
