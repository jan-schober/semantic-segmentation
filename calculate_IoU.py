import glob

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    target_list = sorted(glob.glob('/home/schober/vkitti_conv/labels_color_city/*.png'))
    prediciton_rendered = sorted(
        glob.glob('/home/schober/semantic-segmentation/logs/rendered_vkitti_segmented/*prediction.png'))
    prediction_augmented = sorted(
        glob.glob('/home/schober/semantic-segmentation/logs/augmented_vkitti_segmented/*prediction.png'))
    prediction_real = sorted(
        glob.glob('/home/schober/semantic-segmentation/logs/real_vkitti_segmented/*prediction.png'))

    # assert len(target_list) == len(prediciton_rendered) == len(prediction_augmented) == len(prediction_real), "Different length of lists"
    #assert len(target_list) == len(prediciton_rendered) == len(prediction_augmented), "Different length of lists"
    iou_arr_rend = []
    iou_arr_real = []
    iou_arr_aug = []

    #for target, pred_real, pred_rend, pred_aug in zip(target_list, prediction_real, prediciton_rendered,
     #                                                 prediction_augmented):
        # for target, pred_rend, pred_aug in zip(target_list, prediciton_rendered, prediction_augmented):
        #iou_real = calculate_iou(target, pred_real)
        #iou_arr_real.append(iou_real)
        #iou_rend = calculate_iou(target, pred_rend)
        #iou_arr_rend.append(iou_rend)
    for target, pred_aug in zip(target_list, prediction_augmented):
        iou_aug = calculate_iou(target, pred_aug)
        iou_arr_aug.append(iou_aug)

    pd.options.display.max_colwidth = 150
    #iou_pd_rend = pd.DataFrame({'iou': iou_arr_rend, 'path_rendered': prediciton_rendered})
    #iou_pd_rend_sort = iou_pd_rend.sort_values(by=['iou'], ascending=True)
    #print(iou_pd_rend_sort.head(25))
    iou_pd_aug = pd.DataFrame({'iou': iou_arr_aug, 'path_augmented': prediction_augmented})
    iou_pd_aug_sort = iou_pd_aug.sort_values(by=['iou'], ascending=True)
    print(iou_pd_aug_sort.head(25))
    #iou_pd_real = pd.DataFrame({'iou': iou_arr_real, 'path_real': prediction_real})
    #iou_pd_real_sort = iou_pd_real.sort_values(by=['iou'], ascending=True)
    #print(iou_pd_real_sort.head(25))

    #mean_real = np.mean(iou_arr_real)
    #m_real_str = "{:.3}".format(mean_real)
    #mean_rend = np.mean(iou_arr_rend)
    #m_rend_str = "{:.3}".format(mean_rend)
    mean_aug = np.mean(iou_arr_aug)
    m_aug_str = "{:.3}".format(mean_aug)

    #plt.plot(iou_arr_real, label='iou-real', lw=0.5)
    #plt.plot(iou_arr_rend, label='iou-rend', lw=0.5)
    plt.plot(iou_arr_aug, label='iou-aug', lw=0.5)
    plt.xlabel('images')
    plt.ylabel('iou-score')
    plt.legend(loc="lower right")
    #plt.title('Mean real = ' + m_real_str + ' Mean rendered = ' + m_rend_str + ' Mean augmented = ' + m_aug_str)
    #plt.title(' Mean rendered = ' + m_rend_str + ' Mean augmented = ' + m_aug_str)
    plt.title( 'Mean augmented = ' + m_aug_str)
    plt.savefig('iou_arr.png', dpi=500)


def calculate_iou(target_path, prediction_path):
    # alle cityscapes rgbs die nicht in vkitti vorkommen
    irrelevant_rgb = [[111, 74, 0], [81, 0, 81], [244, 35, 232], [250, 170, 160], [230, 150, 140], [102, 102, 156],
                      [190, 153, 153], [150, 100, 100], [150, 120, 90],
                      [153, 153, 153], [220, 20, 60], [255, 0, 0], [0, 0, 90], [0, 0, 110], [0, 80, 100], [0, 0, 230],
                      [119, 11, 32]]

    target = imageio.imread(target_path)
    prediction = imageio.imread(prediction_path)
    prediction = prediction[:, :, :3]


    ##  unlabeled und misc von gt (target) auch in pred weiß machen
    unlabeled_mask = (target == [0,0,0]).all(axis=2)
    prediction[unlabeled_mask] = [0,0,0]

    ## fuer alle irrelevanten rgbs in predicition, prediction und target weiß machen
    for rgb in irrelevant_rgb:
        irrel_mask = (prediction == rgb).all(axis=2)
        prediction[irrel_mask] = [0,0,0]
        target[irrel_mask] = [0,0,0]

    target_w_mask = (target != [0, 0, 0]).all(axis=2)
    pred_w_mask =   (prediction != [0, 0, 0]).all(axis=2)
    white_mask = np.logical_or(target_w_mask, pred_w_mask)

    intersection = np.logical_and(target[white_mask], prediction[white_mask])
    union = np.logical_or(target[white_mask], prediction[white_mask])

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


if __name__ == '__main__':
    main()
