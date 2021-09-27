import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

'''
This script creates the video which i showed at my presentation
'''

# The file name for the exported video
video_name = 'iou_keras.avi'

'''
Folders for the rendered Images, augmented images, ground truth semantic segmentatition and predicted semantic segmentation
'''
rend_list= sorted(glob.glob('/home/schober/carla/output_carla_town_01/darknet/data/obj/*.jpg'))
aug_list = sorted(glob.glob('/home/schober/carla/for_yolov5/images/val_augmented/*.png'))
gt_label = sorted(glob.glob('/home/schober/carla/output_carla_town_01/sem_filled/*.png'))
pred_list = sorted(glob.glob('/home/schober/semantic-segmentation/logs/carla_augmented_segmented/*prediction.png'))

# numpy array with the iou values
iou_arr = np.load('iou_arr_keras.npy')

assert len(rend_list) ==  len(aug_list) == len(gt_label) == len(pred_list),  "Different Lenght of lists"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# creates an first image grid so we know the exact resolution for the video
first_list = [rend_list[0],gt_label[0], aug_list[0], pred_list[0]]
image, *images = [Image.open(file) for file in first_list]
example_grid = image_grid([image, *images], rows=2, cols=2)
width, height = example_grid.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

# iteration over the different images from the different folders
for ren_path, aug_path, gt_path, pred_path, iou in zip(rend_list, aug_list, gt_label, pred_list, iou_arr):

    ren_img = Image.open(ren_path)
    aug_img = Image.open(aug_path)
    gt_img = Image.open(gt_path)
    pred_img = Image.open(pred_path).convert('RGB')
    # puts the iou value in the left top corner of the predicted semantic segmented image
    pred_draw = ImageDraw.Draw(pred_img)
    font = ImageFont.load_default()
    pred_draw.text((0, 0), str(iou), (255, 255, 255), font=font)

    # creates an image grid with the 4 images and writes an image with it
    grid = image_grid([ren_img, gt_img, aug_img, pred_img], rows=2, cols=2)
    grid_cv = np.array(grid)
    grid_cv = cv2.cvtColor(grid_cv, cv2.COLOR_RGB2BGR)
    video.write(grid_cv)
video.release()
