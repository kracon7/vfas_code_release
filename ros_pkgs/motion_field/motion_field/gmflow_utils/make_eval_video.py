import os
import cv2
import argparse
import numpy as np
from glob import glob
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_dir', '-d', default='tmp', type=str,
                        help='where to load all images')
    parser.add_argument('--output', '-o', default='tmp.avi', type=str,
                        help='where to save video')
    
    return parser.parse_args()


def main():

    args = get_args_parser()
    data_dir = args.data_dir
    color_filenames = sorted(glob(os.path.join(data_dir, 'color') + '/*'))
    depth_filenames = sorted(glob(os.path.join(data_dir, 'depth') + '/*'))
    mag_pred_filenames = sorted(glob(os.path.join(data_dir, 'flow_mag_pred') + '/*'))
    rad_pred_filenames = sorted(glob(os.path.join(data_dir, 'flow_rad_pred') + '/*'))
    mag_gt_filenames = sorted(glob(os.path.join(data_dir, 'flow_mag_gt') + '/*'))
    rad_gt_filenames = sorted(glob(os.path.join(data_dir, 'flow_rad_gt') + '/*'))

    img = np.load(color_filenames[0])
    im_h, im_w = img.shape[0], img.shape[1]

    frameSize = (2*im_w, 3*im_h)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), 5, frameSize)

    for i in range(len(color_filenames) - 2):
        print("Processing frame %s"%color_filenames[i])
        num = str(int(color_filenames[i].split('_')[-1].split('.')[0]))
        print(color_filenames[i], depth_filenames[i], mag_pred_filenames[i], rad_pred_filenames[i], mag_gt_filenames[i], rad_gt_filenames[i])
        if not (num in color_filenames[i] and num in depth_filenames[i] and
                num in mag_pred_filenames[i] and num in rad_pred_filenames[i] and
                num in mag_gt_filenames[i] and num in rad_gt_filenames[i]):
            raise Exception('Frame idx inconsistant!!!')

        color = np.load(color_filenames[i])
        depth = np.load(depth_filenames[i]).astype('float')
        depth = (255* (depth / depth.max())).astype('uint8')
        depth = np.tile(depth, (1,1,3))

        mag_pred = np.asarray(Image.open(mag_pred_filenames[i]))
        rad_pred = np.asarray(Image.open(rad_pred_filenames[i]))
        mag_gt = np.asarray(Image.open(mag_gt_filenames[i]))
        rad_gt = np.asarray(Image.open(rad_gt_filenames[i]))
        canvas = np.zeros((3*im_h, 2*im_w, 3), dtype='uint8')
        canvas[:, :im_w, :] = np.concatenate([color, mag_pred, rad_pred], axis=0)
        canvas[:,im_w:,:] = np.concatenate([depth, mag_gt, rad_gt], axis=0)
        out.write(canvas)

    out.release()

if __name__ == '__main__':
    main()