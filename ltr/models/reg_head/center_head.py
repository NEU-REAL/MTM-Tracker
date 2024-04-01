import torch.nn as nn
import numpy as np
from .center_head_template import CenterHeadTemplate

class CenterHead(CenterHeadTemplate):
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, voxel_size):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels, grid_size=grid_size, 
                        point_cloud_range=point_cloud_range, voxel_size=voxel_size)

        self.z_layers = self.make_final_layers(input_channels, self.model_cfg.Z_FC)
        self.ry_layers = self.make_final_layers(input_channels, self.model_cfg.RY_FC)
        self.whl_layers = self.make_final_layers(input_channels, self.model_cfg.WHL_FC)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.z_layers[-1].weight, mean=0, std=0.001)
        nn.init.normal_(self.ry_layers[-1].weight, mean=0, std=0.001)

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)

        batch_size = gt_boxes.shape[0]
        local_gt = gt_boxes.clone().squeeze(1)
        real_height = local_gt.clone()[:,2]
        center_offset = input_dict['center_offset']
        local_gt[:,:3] -= center_offset
        local_gt = local_gt.unsqueeze(1)

        search_points = input_dict['search_points']
        targets_dict = self.assign_stack_targets(local_gt, real_height, search_points)

        return targets_dict

    def forward(self, batch_dict):
        self.feature_map_stride = batch_dict['spatial_features_stride']
        fusion_feature = batch_dict['fusion_feature']

        z_preds = self.z_layers(fusion_feature)
        ry_preds = self.ry_layers(fusion_feature)
        whl_preds = self.whl_layers(fusion_feature)

        z_preds = z_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        ry_preds = ry_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        whl_preds = whl_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        batch_dict['z_preds'] = z_preds
        batch_dict['ry_preds'] = ry_preds
        batch_dict['whl_preds'] = whl_preds

        ret_dict = {
                    'cls_preds': batch_dict['cls_preds'],
                    'z_preds': z_preds,
                    'ry_preds': ry_preds,
                    'whl_preds': whl_preds,
                    'motion_preds': batch_dict['motion_preds']
                    }

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['cls_labels'] = targets_dict['cls_labels']
            ret_dict['reg_labels'] = targets_dict['reg_labels']
            ret_dict['reg_mask'] = targets_dict['reg_mask']
            ret_dict['ind_labels'] = targets_dict['ind_labels']
            ret_dict['object_dim'] = batch_dict['object_dim']
            ret_dict['motion_map'] = batch_dict['motion_map']
            ret_dict['motion_mask'] = batch_dict['motion_mask']
            ret_dict['corner_loss'] = batch_dict['corner_loss']
            self.forward_ret_dict = ret_dict

        return batch_dict

        