from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from .gmflow_utils import frame_utils
from .gmflow_utils.flow_viz import save_vis_flow_tofile
from .gmflow_utils.utils import InputPadder, compute_out_of_boundary_mask
from glob import glob
from .gmflow.geometry import forward_backward_consistency_check


class FlowPredictor:
    def __init__(self, model, 
                 inference_size,
                 padding_factor,
                 attn_splits_list=None,
                 corr_radius_list=None,
                 prop_radius_list=None,
                 pred_bidir_flow=False,):
        self.model = model
        self.inference_size = inference_size
        self.padding_factor = padding_factor
        self.attn_splits_list = attn_splits_list
        self.corr_radius_list = corr_radius_list
        self.prop_radius_list = prop_radius_list
        self.pred_bidir_flow = pred_bidir_flow

    def pred(self, img_1: torch.tensor, img_2: torch.tensor, mask: torch.tensor):
        
        if self.inference_size is None:
            padder = InputPadder(img_1.shape, padding_factor=self.padding_factor)
            img_1, img_2 = padder.pad(img_1[None].cuda(), img_2[None].cuda())
        else:
            img_1, img_2 = img_1[None].cuda(), img_2[None].cuda()

        # resize before inference
        if self.inference_size is not None:
            assert isinstance(self.inference_size, list) or isinstance(self.inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=self.inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=self.inference_size, mode='bilinear',
                                   align_corners=True)

        with torch.no_grad():
            now = time.time()
            results_dict = self.model(
                                img_1, img_2,
                                attn_splits_list=self.attn_splits_list,
                                corr_radius_list=self.corr_radius_list,
                                prop_radius_list=self.prop_radius_list,
                                pred_bidir_flow=self.pred_bidir_flow,
                                )
            print("Inference time: %.2fms"%((time.time() - now) * 1e3))
            flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
            flow = flow_pr[0].permute(1, 2, 0)  # [H, W, 2]
        if mask is not None:
            flow[mask] = 0
        return flow


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        rgba = self.scalarMap.to_rgba(val)
        return rgba

@torch.no_grad()
def inference_on_dir(model,
                     inference_dir,
                     output_path='output',
                     padding_factor=8,
                     inference_size=None,
                     paired_data=False,  # dir of paired testdata instead of a sequence
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     pred_bidir_flow=False,
                     fwd_bwd_consistency_check=False,
                     mask=None
                     ):
    """ Inference on a directory """
    model.eval()

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    os.makedirs(output_path, exist_ok=True)

    raw_dir = os.path.join(os.path.dirname(output_path), 'raw_flow') 
    os.makedirs(raw_dir, exist_ok=True)

    filenames = sorted(glob(inference_dir + '/*'))
    print('%d images found' % len(filenames))

    stride = 2 if paired_data else 1

    if paired_data:
        assert len(filenames) % 2 == 0

    for test_id in range(0, len(filenames) - 1, stride):

        image1 = frame_utils.read_gen(filenames[test_id])
        image2 = frame_utils.read_gen(filenames[test_id + 1])

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

        if inference_size is None:
            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        else:
            image1, image2 = image1[None].cuda(), image2[None].cuda()

        # resize before inference
        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)

        now = time.time()
        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             pred_bidir_flow=pred_bidir_flow,
                             )
        print("Frame %d, inference time: %.2fms"%(test_id, (time.time() - now) * 1e3))

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if inference_size is None:
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        else:
            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        output_file = os.path.join(output_path, 
                                   os.path.basename(filenames[test_id])[:-4] + '_flow.png')

        # Apply finger mask
        if mask is not None:
            flow[mask] = 0

        # save flow visualization
        save_vis_flow_tofile(flow, output_file)

        # save raw flow
        flow_output_file = os.path.join(raw_dir,
                                        os.path.basename(filenames[test_id])[:-4] + '_flow.npy')
        np.save(flow_output_file, flow)



        # also predict backward flow
        if pred_bidir_flow:
            assert flow_pr.size(0) == 2  # [2, H, W, 2]

            if inference_size is None:
                flow_bwd = padder.unpad(flow_pr[1]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
            else:
                flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_flow_bwd.png')

            # save vis flow
            save_vis_flow_tofile(flow_bwd, output_file)

            # forward-backward consistency check
            # occlusion is 1
            if fwd_bwd_consistency_check:
                if inference_size is None:
                    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
                    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
                else:
                    fwd_flow = flow_pr[0].unsqueeze(0)
                    bwd_flow = flow_pr[1].unsqueeze(0)

                fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)  # [1, H, W] float

                fwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ.png')
                bwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ_bwd.png')

                Image.fromarray((fwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(fwd_occ_file)
                Image.fromarray((bwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(bwd_occ_file)
