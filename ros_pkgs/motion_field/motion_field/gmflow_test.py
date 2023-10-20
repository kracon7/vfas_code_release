import torch
from glob import glob
import argparse
import numpy as np
import os
import cv2
from PIL import Image
from gmflow.gmflow import GMFlow
from gmflow_evaluate import FlowPredictor, MplColorHelper
from gmflow_utils.frame_utils import read_gen

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')

    # resume pretrained model or resume training
    parser.add_argument('--checkpoint', default='', type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    parser.add_argument('--output_dir', '-o', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_raw', action='store_true',
                        help='save the raw flow')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo')
    parser.add_argument('--mask', '-m', default='', type=str,
                        help='where to flow mask')
    parser.add_argument('--di', type=int, default=1,
                        help="number of frame skip")

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)

    return parser


def main(args):

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:{}'.format(args.local_rank))

    # model
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    
    # resume checkpoints
    if args.checkpoint != '':
        print('Load checkpoint: %s' % args.checkpoint)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.checkpoint, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model.load_state_dict(weights, strict=args.strict_resume)

    flow_predictor = FlowPredictor(model,
                                   args.padding_factor,
                                   args.inference_size,
                                   args.attn_splits_list,
                                   args.corr_radius_list,
                                   args.prop_radius_list,
                                   args.pred_bidir_flow)
    
    if args.mask != '':
        finger_mask_tensor = torch.tensor(np.asarray(Image.open(args.mask)),
                                          device=device)
        finger_mask = finger_mask_tensor[:,:,0] > 0

    rad_colormap = MplColorHelper('twilight', -np.pi, np.pi)
    mag_colormap = MplColorHelper('gray', 0, 40)

    flow_mag_dir = os.path.join(args.output_dir, 'flow_mag_pred')
    flow_rad_dir = os.path.join(args.output_dir, 'flow_rad_pred')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(flow_mag_dir, exist_ok=True)
    os.makedirs(flow_rad_dir, exist_ok=True)
    if args.save_raw:
        raw_dir = os.path.join(args.output_dir, 'flow_raw_pred') 
        os.makedirs(raw_dir, exist_ok=True)

    abs_filenames = sorted(glob(args.inference_dir + '/*'))
    rel_filenames = sorted(os.listdir(args.inference_dir))
    print('%d images found' % len(abs_filenames))

    for test_id in range(len(abs_filenames) - args.di):

        image1 = read_gen(abs_filenames[test_id])
        image2 = read_gen(abs_filenames[test_id + args.di])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image, for example, HD1K
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        flow = flow_predictor.pred(image1, image2, finger_mask)

        flow_np = flow.cpu().numpy()
        if args.save_raw:
            np.save(os.path.join(raw_dir, rel_filenames[test_id].split('.')[0] + '.npy'), flow_np)
        
        flow_mag = np.linalg.norm(flow_np, axis=2)
        flow_rad = np.arctan2(flow_np[:,:,0], flow_np[:,:,1])
        print("rad: %.3f, %.3f mag: %.3f, %.3f"%(flow_rad.min(), flow_rad.max(), 
                                                 flow_mag.min(), flow_mag.max()))

        flow_mag_vis = (mag_colormap.get_rgb(flow_mag)[:,:,:3] * 255).astype('uint8')
        flow_rad_vis = (rad_colormap.get_rgb(flow_rad)[:,:,:3] * 255).astype('uint8')
        
        mag_output_path = os.path.join(flow_mag_dir, rel_filenames[test_id])
        rad_output_path = os.path.join(flow_rad_dir, rel_filenames[test_id])
        
        cv2.imwrite(mag_output_path, flow_mag_vis)
        cv2.imwrite(rad_output_path, flow_rad_vis)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
