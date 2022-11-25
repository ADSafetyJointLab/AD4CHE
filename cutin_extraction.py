# this script is used for scenario analysis
import os
import utils
import logging

lateral_offset = 0.375
logger = logging.getLogger(__name__)


class rss_para:
    def __init__(self):
        ''' RSS '''
        self.rho = 0.75
        self.max_acc = 3
        self.min_decel = 6
        self.max_decel = 6

    def d_long(self, v_r, v_f):
        d_min = v_r * self.rho + 1 / 2 * self.max_acc * pow(self.rho, 2) + pow(v_r + self.rho * self.max_acc,
                                                                               2) / 2 / self.min_decel - pow(v_f,
                                                                                                             2) / 2 / self.max_decel
        return max(d_min, 0)


class ScenarioExtraction(rss_para):
    def __init__(self, dataset_path):
        self.vy = None
        self.y = None
        self.tracks_meta = None
        self.obj_class = None
        self.vx = None
        self.x = None
        self.width = None
        self.height = None
        self.frame = None
        self.followingId = None
        self.laneid = None
        self.ego_offset = None
        self.tracks_dict = None
        self.folder_cutin_id = None
        self.dataset_path = dataset_path
        self.car_pairs = {}
        self.overlap_car_pairs = {}
        self.noncar_pairs = {}
        self.nonoverlap_car_pairs = {}
        self.lanechange = {}

        super(ScenarioExtraction, self).__init__()

    def read_data(self):
        folder_num = len(os.listdir(self.dataset_path))  #
        for i in range(1, folder_num + 1):
            prefix = 'DJI_'
            num_folder = str(i).zfill(4)
            folder_name = prefix + num_folder
            num_file = str(i).zfill(2)
            tracks_name = num_file + '_tracks.csv'
            tracks_meta_name = num_file + '_tracksMeta.csv'
            tracks_path = os.path.join(self.dataset_path, folder_name, tracks_name)
            tracks_meta_path = os.path.join(self.dataset_path, folder_name, tracks_meta_name)
            tracks_labels, self.tracks_dict = utils.load_tracks(tracks_path)
            tracks_meta_labels, self.tracks_meta = utils.load_tracks_meta(tracks_meta_path)

            # get index of parameters
            numlanechanges = utils.get_label_inx(tracks_meta_labels, 'numLaneChanges')
            self.obj_class = utils.get_label_inx(tracks_meta_labels, 'class')

            self.ego_offset = utils.get_label_inx(tracks_labels, 'ego_offset')
            self.laneid = utils.get_label_inx(tracks_labels, 'laneId')
            self.followingId = utils.get_label_inx(tracks_labels, 'followingId')
            self.frame = utils.get_label_inx(tracks_labels, 'frame')
            self.width = utils.get_label_inx(tracks_labels, 'width')
            self.height = utils.get_label_inx(tracks_labels, 'height')
            self.x = utils.get_label_inx(tracks_labels, 'x')
            self.y = utils.get_label_inx(tracks_labels, 'y')
            self.vx = utils.get_label_inx(tracks_labels, 'xVelocity')
            self.vy = utils.get_label_inx(tracks_labels, 'yVelocity')

            # check is lane changing occurs
            for track_meta in self.tracks_meta:
                # number of lane changes, only one lane change
                if int(track_meta[numlanechanges]) == 1:
                    vech_id_meta = int(track_meta[0])
                    cut_in_pairs, is_overlap, is_ego_car = self.find_cutin(vech_id_meta)
                    self.folder_cutin_id = folder_name + '_' + track_meta[0]
                    # some cases have numLaneChange > 0, but ego offset is rather small, return []
                    if not len(cut_in_pairs):
                        continue

                    # car cuts in a car
                    if track_meta[self.obj_class] == 'car' and is_ego_car:
                        # some cases only cut-in vehicle is found, return length 1, recording lane changing
                        self.lanechange[self.folder_cutin_id] = cut_in_pairs
                        # recording only cut-in
                        if len(cut_in_pairs) == 3:
                            self.car_pairs[self.folder_cutin_id] = cut_in_pairs
                            if is_overlap:
                                self.overlap_car_pairs[self.folder_cutin_id] = cut_in_pairs
                            else:
                                self.nonoverlap_car_pairs[self.folder_cutin_id] = cut_in_pairs
                    # car cuts in a noncar, or a noncar cuts in a car
                    elif len(cut_in_pairs) == 3:
                        self.noncar_pairs[self.folder_cutin_id] = cut_in_pairs

                    logger.info(self.folder_cutin_id)

    def find_cutin(self, vech_id_meta):
        cut_in_pairs = []
        t5 = None
        t1 = None
        t3 = None
        ego_id = None
        is_overlap = False
        v_f = None
        cut_in_width = None
        cut_in_x_t1 = None
        cut_in_x_t3 = None
        t1_inx = None
        t5_inx = None
        t3_inx = None
        is_t1_found = is_t5_found = False
        # find cut-in vehicle
        for vech_id, vech_track in self.tracks_dict.items():
            if int(vech_id) == vech_id_meta:
                lane_id_t1 = None
                offset_temp = None
                t3_inx_temp = None
                # define when lane id changes
                for inx in range(len(vech_track)-1):
                    current_laneid = int(vech_track[inx][self.laneid])
                    next_landid = int(vech_track[inx+1][self.laneid])
                    # skip the vehicles on entry and exit lanes
                    if current_laneid > 50 or current_laneid == 0:
                        continue
                    # determine if t3 found, lane id is changed
                    if current_laneid != next_landid:
                        t3_inx_temp = inx
                        offset_temp = float(vech_track[inx][self.ego_offset])
                        break
                # cannot find t3, skip this object
                if offset_temp is None:
                    break
                # determine t3, a car is adjacent to a lane marking
                for inx in range(t3_inx_temp, -1, -1):
                    offset_current = float(vech_track[inx][self.ego_offset])
                    cut_in_height = float(vech_track[inx][self.height])
                    if abs(offset_temp - offset_current) >= cut_in_height/2:
                        t3_inx = inx
                        t3 = vech_track[inx][self.frame]
                        v_f = float(vech_track[inx][self.vx])
                        cut_in_x_t3 = float(vech_track[inx][self.x])
                        break
                # some case lane chang exist, but no offset is within the threshold, e.g., DJI0001, vechid 130
                if t3_inx is None:
                    continue

                # determine t1 - lane changing starts
                for inx in range(t3_inx, 0, -1):
                    offset_current = float(vech_track[inx][self.ego_offset])
                    offset_previous = float(vech_track[inx - 1][self.ego_offset])
                    if abs(offset_current) <= lateral_offset and abs(offset_previous) <= lateral_offset:
                        t1_inx = inx
                        t1 = vech_track[t1_inx][self.frame]
                        lane_id_t1 = int(vech_track[t1_inx][self.laneid])
                        cut_in_width = float(vech_track[t1_inx][self.width])
                        cut_in_x_t1 = float(vech_track[t1_inx][self.x])
                        is_t1_found = True
                        break

                # determine t5, when lane changing ends
                for inx in range(t3_inx, len(vech_track)-1):
                    offset_current = float(vech_track[inx][self.ego_offset])
                    offset_next = float(vech_track[inx + 1][self.ego_offset])
                    # determine when car enters the wandering zone in ego's lane
                    if is_t1_found and abs(offset_current) <= lateral_offset and abs(
                            offset_next) <= lateral_offset:
                        lane_id_t5 = int(vech_track[inx][self.laneid])
                        # abandon the scenarios in which ego goes back to previous lane with interrupting lane changing
                        if lane_id_t5 != lane_id_t1:
                            t5_inx = inx
                            t5 = vech_track[inx][self.frame]
                            ego_id = int(vech_track[inx][self.followingId])
                            is_t5_found = True
                            break

                # save cut-in vehicle data
                if is_t1_found and is_t5_found:
                    cut_in_pairs.append(vech_track[t1_inx:t5_inx])
                    logger.info(f"t1: {t1}, t3: {t3}, t5: {t5}")
                    break

        is_ego_car = False
        if is_t1_found and is_t5_found:
            # find ego vehicle
            for vech_id, vech_track in self.tracks_dict.items():
                if int(vech_id) != ego_id:
                    continue
                frames = [x[self.frame] for x in vech_track]
                # in some cases, the ego appears later than cut-in vehicle these cases should be filtered
                if int(t1) >= int(frames[0]) and int(t5) <= int(frames[-1]):
                    begin_inx = frames.index(t1)
                    end_inx = frames.index(t5)
                    middle_inx = frames.index(t3)
                    ego_width = float(vech_track[begin_inx][self.width])
                    ego_x_t1 = float(vech_track[begin_inx][self.x])
                    ego_x_t3 = float(vech_track[middle_inx][self.x])
                    v_r = float(vech_track[middle_inx][self.vx])

                    # in some cases, ego changes lane during the cut-in, these cases should be filtered
                    lane_id_begin = int(vech_track[begin_inx][self.laneid])
                    lane_id_end = int(vech_track[end_inx][self.laneid])
                    if lane_id_end == lane_id_begin:
                        # check if cut-in or lane change by using rss
                        driving_direction = utils.driving_direction(self.vx, vech_track)
                        if driving_direction == -1:
                            d_x_t1 = ego_x_t1 - cut_in_x_t1
                        elif driving_direction == 1:
                            d_x_t1 = cut_in_x_t1 - ego_x_t1
                        # d_x_t1 = abs(cut_in_x_t1 - ego_x_t1) - (ego_width + cut_in_width) / 2
                        d_x_t3 = abs(cut_in_x_t3 - ego_x_t3) - (ego_width + cut_in_width) / 2
                        d_rss = self.d_long(abs(v_r), abs(v_f))
                        if d_x_t3 <= d_rss and d_x_t1 >= - (ego_width + cut_in_width) / 2:
                            cut_in_pairs.append(vech_track[begin_inx:end_inx])
                            # save the key timestamps
                            cut_in_pairs.append([t1, t3, t5])
                            # check if overlap cut-in occurs
                            if d_x_t1 <= (ego_width + cut_in_width) / 2:
                                is_overlap = True

                break

            # determine if ego is car or not
            for track_meta in self.tracks_meta:
                if int(track_meta[0]) == ego_id:
                    if track_meta[self.obj_class] == 'car':
                        is_ego_car = True
                    break

        return cut_in_pairs, is_overlap, is_ego_car


def main():
    utils.setup_logging()
    # define the dataset path
    dataset_path = '/home/cheng/work/AD4CHE_V1.0/AD4CHE_Data_V1.0'
    save_extracted_data = True

    sce_extra = ScenarioExtraction(dataset_path)
    sce_extra.read_data()

    # print information
    logger.info(f"total lane change scenarios are {len(sce_extra.lanechange)}")
    logger.info(f"total lane cut-in are {len(sce_extra.car_pairs)}")
    logger.info(f"total lane overlap cut-in are {len(sce_extra.overlap_car_pairs)}")
    logger.info(f"total lane non-overlap cut-in are {len(sce_extra.nonoverlap_car_pairs)}")

    # save the object avoid repeating computing
    if save_extracted_data:
        utils.save_object(sce_extra)


if __name__ == '__main__':
    main()
