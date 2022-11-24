import cv2
import numpy as np
import warnings


class lane_info:
    def __init__(self):
        self.current_contour = None
        self.laneidlist = None
        self.contourlist = None
        self.pixel_length = 0.0375
        self.lane_img = None

    def load_img(self, img_path):
        self.lane_img = cv2.imread(img_path, 0)

    def find_contours(self):
        new_lane_img, self.contourlist = self.image_process()
        self.laneidlist, lanes_right2left = self.getIDs(new_lane_img)

    def image_process(self):
        contourlist = []
        img = self.lane_img
        # fill the gap between lanes
        for m in range(len(img)):
            for n in range(len(img[0])):
                if img[m, n] == 0 and 0 < m < 2159:
                    if img[m - 1, n] != 0 and img[m + 1, n] != 0:
                        img[m, n] = img[m - 1, n]

        for grey in range(max(np.unique(img)) + 1):
            ret, thresh1 = cv2.threshold(img, grey, 255, cv2.THRESH_BINARY)
            ret, thresh2 = cv2.threshold(img, grey - 1, 255, cv2.THRESH_BINARY_INV)
            thre = 255 - (thresh1 + thresh2)
            h = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contourlist.append(h)

        return img, contourlist

    @staticmethod
    def getIDs(img):
        numofgrey = np.unique(img)
        goodlanedata = []
        badlanedata = []
        goodlaneID = []
        badlaneID = []
        s = []
        listid = [0 for x in range(0, 25)]
        for i in numofgrey[1:None]:
            ret, thresh1 = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)
            ret, thresh2 = cv2.threshold(img, i - 1, 255, cv2.THRESH_BINARY_INV)
            thre = 255 - (thresh1 + thresh2)
            contourdata = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            m0 = contourdata[0][0][:, 0, :]
            # goodlanes equal main lanes, bad lanes equal lane exit or entry
            if np.array(np.where(m0[:, 0] == 1900)).size > 0 and np.var(m0[:, 1]) < 5000 and 1750 > np.mean(
                    m0[:, 1]) > 500:
                goodlanedata.append(np.mean(m0[:, 1]))
                goodlaneID.append(i)
            else:
                badlanedata.append(np.mean(m0[:, 1]))
                badlaneID.append(i)
        for i in goodlaneID:
            listid[i] = len(goodlaneID) - list(np.array(goodlanedata).argsort()).index(goodlaneID.index(i))
        for i in badlaneID:
            listid[i] = 100 + len(badlaneID) - list(np.array(badlanedata).argsort()).index(badlaneID.index(i))
        pos = (n for n in listid if n > 100)
        for x in pos:
            s.append(x)
        s.append((len(goodlaneID) // 2) + 1)
        lanes_right2left = {"toRight": list(range(1, (len(goodlaneID) // 2) + 1)),
                            "toLeft": list(range((len(goodlaneID) // 2) + 2, len(goodlaneID) + 1)), "NaN": s}

        return listid, lanes_right2left

    def exceed_lane_marking(self, lane_id, pos_x, pos_y, vech_height):
        # skip the vehicles on entry and exit lanes
        if lane_id > 50 or lane_id == 0:
            return None

        current_lane = np.where(np.array(self.laneidlist) == lane_id)
        current_contour = self.contourlist[current_lane[0][0]]
        warnings.filterwarnings('ignore')

        current_contour_x = current_contour[0][0][:, 0, 0]
        pos_x_pixel = int(pos_x / self.pixel_length)
        array = np.asarray(current_contour_x)
        idx = (np.abs(array - pos_x_pixel)).argmin()
        x_pointer = np.where(array == array[idx])

        y1 = current_contour[0][0][x_pointer[0][0], 0, 1] * self.pixel_length
        y2 = current_contour[0][0][x_pointer[0][1], 0, 1] * self.pixel_length

        lower_marking = max(y2, y1)
        upper_marking = min(y2, y1)
        if (pos_y - upper_marking) <= vech_height/2:
            return "upper"
        elif (lower_marking - pos_y) <= vech_height/2:
            return "lower"
        else:
            return None
        # return abs(y1 - y2) * self.pixel_length
