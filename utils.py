# this script is used for defining common used functions
import csv
import pickle
import logging
import sys

import numpy as np


def get_label_inx(labels, label):
    return labels.index(label)

def driving_direction(vx_indx, tracks_dict):
    vx = [float(x[vx_indx]) for x in tracks_dict]
    if np.average(vx) < 0:
        # heading to the left
        heading = -1
    else:
        # heading to the right
        heading = 1
    return heading

def load_tracks(tracks_path):
    vech_data = []
    tracks_dict = {}
    with open(tracks_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                tracks_labels = row
                line_count += 1
                id_inx = get_label_inx(tracks_labels, 'id')
            # create dict for saving data according to vehicle id
            else:
                # will execute only once
                if line_count == 1:
                    old_vech_id = row[id_inx]
                    vech_data.append(row)
                else:
                    new_vech_id = row[id_inx]
                    if new_vech_id != old_vech_id:
                        tracks_dict[old_vech_id] = vech_data
                        vech_data = []
                        old_vech_id = new_vech_id
                    vech_data.append(row)

                line_count += 1
    return tracks_labels, tracks_dict


def load_tracks_meta(tracks_meta_path):
    tracks_meta = []
    with open(tracks_meta_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                tracks_meta_labels = row
                line_count += 1
            else:
                tracks_meta.append(row)
                line_count += 1
    return tracks_meta_labels, tracks_meta


def setup_logging():
    # Add %(asctime)s  for time
    log_formatter = logging.Formatter("[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def save_object(obj):
    try:
        with open("data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
