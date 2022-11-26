from cutin_extraction import ScenarioExtraction
import utils
from parameter_distributions import ParameterDistribution
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"


class BehaviorAnalysis:
    def __init__(self,
                 frame: int,
                 width: int,
                 vx: int,
                 vy: int,
                 x: int,
                 ego_offset: int, ):

        self.frame = frame
        self.width = width
        self.vx = vx
        self.vy = vy
        self.x = x
        self.ego_offset = ego_offset

        self.car_pairs_constant_offset = {}
        self.car_pairs_changing_offset = {}

    @staticmethod
    def trenddetector(list_of_index, array_of_data, order=1):
        result = np.polyfit(list_of_index, list(array_of_data), order)
        slope = result[-2]
        return float(slope)

    def plot_ego_offset(self, overlap_pairs):
        fig, ax = plt.subplots(3)
        thres = 0.02

        offset_t1_t3_p = []
        time_t1_t3_p = []
        offset_t1_t3_n = []
        time_t1_t3_n = []
        offset_t1_t3_m = []
        time_t1_t3_m = []

        increasing_offset = {}
        decreasing_offset = {}
        for key, car_pair in overlap_pairs.items():
            offset_t1_t3 = []
            time_t1_t3 = []

            ego = car_pair[1]
            key_timestamps_t1 = int(car_pair[2][0])
            key_timestamps_t3 = int(car_pair[2][1])

            for ego_data in ego:
                if key_timestamps_t1 <= int(ego_data[self.frame]) <= key_timestamps_t3:
                    offset = float(ego_data[self.ego_offset])
                    time = (int(ego_data[self.frame]) - key_timestamps_t1) / 30
                    offset_t1_t3.append(offset)
                    time_t1_t3.append(time)
                elif int(ego_data[self.frame]) > key_timestamps_t3:
                    break
            if len(offset_t1_t3) < 2:
                continue

            slope = self.trenddetector(time_t1_t3, offset_t1_t3)

            if slope > thres:
                offset_t1_t3_p.append(offset_t1_t3)
                time_t1_t3_p.append(time_t1_t3)
                increasing_offset[key] = car_pair
            elif slope < -thres:
                offset_t1_t3_n.append(offset_t1_t3)
                time_t1_t3_n.append(time_t1_t3)
                decreasing_offset[key] = car_pair
            else:
                offset_t1_t3_m.append(offset_t1_t3)
                time_t1_t3_m.append(time_t1_t3)
                self.car_pairs_constant_offset[key] = car_pair

        for inx in range(len(offset_t1_t3_p)):
            ax[0].plot(time_t1_t3_p[inx][::4], offset_t1_t3_p[inx][::4], color='grey')
        for inx in range(len(offset_t1_t3_n)):
            ax[1].plot(time_t1_t3_n[inx][::4], offset_t1_t3_n[inx][::4], color='#41A96E')
        for inx in range(len(offset_t1_t3_m)):
            ax[2].plot(time_t1_t3_m[inx][::4], offset_t1_t3_m[inx][::4], color='#A57ED0')
        for ax_ in ax:
            ax_.grid(alpha=0.75)
            ax_.set_xlim([0, 15])

        # plot the bar
        plt.subplots(1)
        height = [len(offset_t1_t3_p), len(offset_t1_t3_n), len(offset_t1_t3_m)]
        bars = ('Increasing', 'Decreasing', 'Constant')
        x_pos = np.arange(len(bars))

        # Create bars with different colors
        bar1 = plt.bar(x_pos, height, color=['#545454', '#41A96E', '#A57ED0'])

        # Create names on the x-axis
        plt.xticks(x_pos, bars)
        plt.grid(axis='y', alpha=0.75)
        for bar in bar1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height), ha='center')

        self.car_pairs_changing_offset = increasing_offset | decreasing_offset
        self.plot_dx0()

    def get_vy_dx(self, car_pairs):
        rel_dx0 = []
        vy0 = []
        for key, car_pair in car_pairs.items():
            cutin_car = car_pair[0]
            ego = car_pair[1]
            key_timestamps_t1 = int(car_pair[2][0])
            key_timestamps_t3 = int(car_pair[2][1])
            rel_d_x = None

            for inx, car in enumerate(ego):
                if int(car[self.frame]) == key_timestamps_t3:
                    inx_ = inx
                    break

            width = float(cutin_car[inx_][self.width]) / 2 + float(ego[inx_][self.width]) / 2
            if utils.driving_direction(self.vx, cutin_car) == 1:
                rel_d_x = float(cutin_car[inx_][self.x]) - float(ego[inx_][self.x]) - width
            elif utils.driving_direction(self.vx, cutin_car) == -1:
                rel_d_x = float(ego[inx_][self.x]) - float(cutin_car[inx_][self.x]) - width
            if rel_d_x is not None:
                rel_dx0.append(rel_d_x)

            vy = abs(float(cutin_car[inx_][self.vy]))
            vy0.append(vy)

        return rel_dx0, vy0

    def plot_dx0(self):
        rel_dx0_changing, vy0_changing = self.get_vy_dx(self.car_pairs_changing_offset)
        rel_dx0, vy0 = self.get_vy_dx(self.car_pairs_constant_offset)
        plt.subplots(1)
        plt.scatter(rel_dx0, vy0, color='red')
        plt.scatter(rel_dx0_changing, vy0_changing, color='blue')


def main():
    obj = utils.load_object("data.pickle")
    overlap_car_pairs = obj.overlap_car_pairs
    behavior_data = BehaviorAnalysis(obj.frame, obj.width, obj.vx, obj.vy, obj.x, obj.ego_offset)
    behavior_data.plot_ego_offset(overlap_car_pairs)

    plt.show()


if __name__ == '__main__':
    main()
