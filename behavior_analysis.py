from cutin_extraction import ScenarioExtraction
import utils
from parameter_distributions import ParameterDistribution
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib

plt.rcParams["font.family"] = "Times New Roman"


class BehaviorAnalysis:
    def __init__(self,
                 frame: int,
                 width: int,
                 height: int,
                 vx: int,
                 vy: int,
                 x: int,
                 y: int,
                 ego_offset: int,
                 accx: int):

        self.frame = frame
        self.width = width
        self.height = height
        self.vx = vx
        self.vy = vy
        self.x = x
        self.y = y
        self.ego_offset = ego_offset
        self.accx = accx

        self.car_pairs_constant_offset = {}
        self.car_pairs_changing_offset = {}

    @staticmethod
    def trenddetector(list_of_index, array_of_data, order=1):
        result = np.polyfit(list_of_index, list(array_of_data), order)
        return result

    def plot_ego_offset(self, overlap_pairs):
        '''determine whisch overlap scenarios has increasing/decreasing lateral offset, which keeps more or less
        constant '''
        fig, ax = plt.subplots(3)
        thres = 0.01

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
            accx_t3_t5 = []

            ego = car_pair[1]
            key_timestamps_t1 = int(car_pair[2][0])
            key_timestamps_t3 = int(car_pair[2][1])
            key_timestamps_t5 = int(car_pair[2][2])

            for ego_data in ego:
                # should before t3, after that ego has no motivation to evade
                if key_timestamps_t1 <= int(ego_data[self.frame]) <= key_timestamps_t3:
                    offset = float(ego_data[self.ego_offset])
                    time = (int(ego_data[self.frame]) - key_timestamps_t1) / 30
                    offset_t1_t3.append(offset)
                    time_t1_t3.append(time)
                elif key_timestamps_t3 <= int(ego_data[self.frame]) <= key_timestamps_t5:
                    accx = float(ego_data[self.accx])
                    accx_t3_t5.append(accx)
                elif int(ego_data[self.frame]) > key_timestamps_t5:
                    break
            # in some cases, t1 to t3 are equal.
            if len(offset_t1_t3) < 2:
                continue

            slope = float(self.trenddetector(time_t1_t3, offset_t1_t3)[-2])

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

    def get_vy_dy(self, car_pairs):
        ''' calculate inverse TTC and corresponding ego velocity at timestamp t3'''
        rel_dy0 = []
        vy0 = []
        ittc_y = []
        vx0 = []
        for key, car_pair in car_pairs.items():
            cutin_car = car_pair[0]
            ego = car_pair[1]
            key_timestamps_t1 = int(car_pair[2][0])
            key_timestamps_t3 = int(car_pair[2][1])
            key_timestamps_t5 = int(car_pair[2][2])

            for inx, car in enumerate(ego):
                if int(car[self.frame]) == key_timestamps_t3:
                    inx_ = inx
                    break

            height = float(cutin_car[inx_][self.height]) / 2 + float(ego[inx_][self.height]) / 2
            rel_d_y = abs(float(ego[inx_][self.y]) - float(cutin_car[inx_][self.y])) - height

            if rel_d_y is not None:
                rel_dy0.append(rel_d_y)

            vy = abs(float(cutin_car[inx_][self.vy]))
            vy0.append(vy)
            vx = abs(float(cutin_car[inx_][self.vx]))
            vx0.append(vx)

            ittc_y.append(vy / rel_d_y)

        return vx0, ittc_y

    def plot_dx0(self):
        vx0_changing, ittc_y_changing = self.get_vy_dy(self.car_pairs_changing_offset)
        vx0_changing_new = []
        ittc_y_changing_new =[]
        for i in range(len(vx0_changing)):
            if vx0_changing[i] <= 10:
                vx0_changing_new.append(vx0_changing[i])
                ittc_y_changing_new.append(ittc_y_changing[i])
        coef = np.polyfit(vx0_changing_new, ittc_y_changing_new, 1)
        poly1d_changing = np.poly1d(coef)

        vx0, ittc_y = self.get_vy_dy(self.car_pairs_constant_offset)
        coef = np.polyfit(vx0, ittc_y, 1)
        poly1d_const = np.poly1d(coef)

        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.scatter(vx0, ittc_y, color='#FF8D57')
        ax1.scatter(vx0_changing, ittc_y_changing, color='#C0ACA1')
        ax1.grid(alpha=0.7)

        ax2.scatter(vx0, ittc_y, color='#FF8D57')
        ax2.scatter(vx0_changing_new, ittc_y_changing_new, color='#C0ACA1')
        ax2.plot(vx0, poly1d_const(vx0), color='#FF8D57')
        ax2.plot(vx0_changing_new, poly1d_changing(vx0_changing_new), color='#C0ACA1')
        ax2.grid(alpha=0.7)

        fig.text(0.5, 0.03, r'Ego velocity at $T_{3}$ [m/s]', ha='center')
        fig.text(0.04, 0.4, r'Inverse TTC at $T_{3}$ [1/s]', rotation='vertical')



def main():
    obj = utils.load_object("data.pickle")
    overlap_car_pairs = obj.overlap_car_pairs
    behavior_data = BehaviorAnalysis(obj.frame, obj.width, obj.height, obj.vx, obj.vy, obj.x, obj.y, obj.ego_offset, obj.accx)
    behavior_data.plot_ego_offset(overlap_car_pairs)

    plt.show()


if __name__ == '__main__':
    main()
