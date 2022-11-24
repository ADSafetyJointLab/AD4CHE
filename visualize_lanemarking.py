# this script is used visualize the lane markings

import os
import road_geometry
import matplotlib.pyplot as plt
import matplotlib.image as img


def main():
    # define the dataset path
    dataset_path = '/home/cheng/work/AD4CHE_V1.0/AD4CHE_Data_V1.0'
    folder_num = len(os.listdir(dataset_path))

    # test the first image
    i = 1
    prefix = 'DJI_'
    num_folder = str(i).zfill(4)
    folder_name = prefix + num_folder
    num_file = str(i).zfill(2)

    # load and process.png image
    img_name = num_file + '_lanePicture.png'
    img_path = os.path.join(dataset_path, folder_name, img_name)
    lane_contours = road_geometry.lane_info(img_path)
    original_img = img.imread(os.path.join(dataset_path, folder_name, num_file + '_highway.png'))

    # find lane markings
    lane_contours.load_img()
    new_lane_img, contourlist = lane_contours.image_process()
    for contour in contourlist:
        for data in contour[0]:
            points_x = []
            points_y = []
            for point in data:
                points_x.append(point[0][0])
                points_y.append(point[0][1])
            plt.plot(points_x, points_y, color="blue", linewidth=1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.show()
    plt.savefig('figure_0001.png', format='png', dpi=1200)


if __name__ == '__main__':
    main()
