import glob
from collections import namedtuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.metrics import MeanIoU

# folder with the ground truth semantic images
target_list = sorted(glob.glob('/home/schober/carla/output_carla_town_*/sem_filled/*.png'))

# folder with the predicted semantic images
prediction_augmented = sorted(glob.glob('/home/schober/semantic-segmentation/logs/best_images/*/*prediction.png'))

# true if we want a plot with iou over images
plot = True
fig_name = 'iou_final.eps'

# labels from the cityscapes dataset
# the labels with capital letters are added to match with the output of the carla semantic segmentation
Label = namedtuple('Label', [
    'name',
    'id',
    'trainId',
    'category',
    'categoryId',
    'hasInstances',
    'ignoreInEval',
    'color',
])
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    Label('LANE', 7, 0, 'void', 7, False, False, (157, 234, 50)),
    Label('DYNAMIK', 5, 255, 'void', 7, False, False, (170, 120, 50)),
    Label('STATIK', 4, 255, 'void', 7, False, False, (110, 190, 160)),
    Label('TERRAIN', 22, 9, 'void', 7, False, False, (145, 170, 100)),
    Label('FENCE', 13, 4, 'void', 7, False, False, (100, 40, 40)),
    Label('OTHER', 0, 255, 'void', 7, False, False, (55, 90, 80)),
    Label('WATER', 22, 9, 'void', 7, False, False, (45, 60, 150)),
]


def main():
    num_classes = 20
    iou_arr = []
    iou_dict = dict()

    for target, prediction in zip(target_list, prediction_augmented):

        # reads the target and prediciton image and converts it to trainID grayscale image
        target_img = cv2.imread(target, cv2.COLOR_BGR2RGB)
        target_img = convert_color_id(target_img[:, :, :3])
        prediction_img = cv2.imread(prediction, cv2.COLOR_BGR2RGB)
        prediction_img = convert_color_id(prediction_img[:, :, :3])

        # put the ignore label on the exact same positions in target and prediction image
        target_img, prediction_img = ignore_label(target_img, prediction_img)

        # reshape the images for keras
        target_img = target_img.reshape(-1)
        prediction_img = prediction_img.reshape(-1)

        # calculates the iou per imagepair
        iou_keras = MeanIoU(num_classes=num_classes)
        iou_keras.update_state(target_img, prediction_img)

        # create conf matrix to ignore the class with id 19
        conf_matrix = np.array(iou_keras.get_weights()).reshape(num_classes, num_classes)
        empty_array = np.empty(num_classes)

        # we calculate iou per class
        for j in range(0, num_classes):
            if (sum(conf_matrix[j, :]) + sum(conf_matrix[:, j]) - conf_matrix[j, j]) == 0:
                empty_array[j] = 0.0
            else:
                empty_array[j] = conf_matrix[j, j] / (
                        sum(conf_matrix[j, :]) + sum(conf_matrix[:, j]) - conf_matrix[j, j])
        # if class 19 is 1.0 we ignore it and calculate the mean iou without it
        if empty_array[19] == 1.0:
            empty_array_0 = empty_array[empty_array != 0]
            iou = (np.sum(empty_array_0) - 1.0) / (len(empty_array_0) - 1)

        # calculation of the mean iou if class 19 is 0
        else:
            empty_array_0 = empty_array[empty_array != 0]
            iou = np.mean(empty_array_0)
        # append the iou of the image and reset the state for new calculation
        iou_arr.append(iou)
        iou_keras.reset_state()

    # calculate the mean iou over all images for plot
    mean_iou = np.mean(iou_arr)
    print('Mean IoU over all Images')
    print(mean_iou)

    if plot:
        m_iou_str = "{:.3}".format(mean_iou)
        plt.plot(iou_arr, lw=0.5)
        plt.grid()
        plt.xlabel('Images')
        plt.ylabel('IoU-Score')
        plt.title('Mean IoU-Score = ' + m_iou_str)
        plt.savefig(fig_name, format='eps', dpi=500)


def convert_color_id(img):
    '''
    The script gets the color semantic segmentation as input and converts the color image to a grayscale image with the
    values from the trainID
    '''
    for label in labels:
        color_rgb = label.color
        color_rgb = [color_rgb[2], color_rgb[1], color_rgb[0]]
        t_id = label.trainId
        color_id = [t_id, t_id, t_id]
        mask = (img == color_rgb).all(axis=2)
        img[mask] = color_id
    return img


def ignore_label(target, prediciton):
    '''
    This function gets the ground truth and predicted semantic segmented image
    In both images it searches for the ignore label and adopts both images so that they have the ignore label
    in the exact same regions, so that in the end we have an iou for the ignore class from 1
    '''
    tar_mask = (target == [255, 255, 255]).all(axis=2)
    pred_mask = (prediciton == [255, 255, 255]).all(axis=2)
    target[pred_mask] = [19, 19, 19]
    target[tar_mask] = [19, 19, 19]
    prediciton[pred_mask] = [19, 19, 19]
    prediciton[tar_mask] = [19, 19, 19]
    return target, prediciton

if __name__ == '__main__':
    main()
