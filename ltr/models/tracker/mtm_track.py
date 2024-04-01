from .track3d_template import Track3DTemplate
import time
import torch
import numpy as np
import copy
class MTM_Track(Track3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.feature_down_sample = model_cfg['POST_PROCESSING']['FEATURE_DOWN_SAMPLE']
        self.motion_down_sample = model_cfg['POST_PROCESSING']['MOTION_DOWN_SAMPLE']
        self.ration = self.feature_down_sample / self.motion_down_sample
        self.voxel_size = model_cfg['POST_PROCESSING']['VOXEL_SIZE']
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # ------ Forward ------
        for cur_module in self.module_list:
            # cur_module_name = cur_module.__class__.__name__
            # start_time = time.time()

            batch_dict = cur_module(batch_dict)

            # end_time = time.time()
            # print(cur_module_name, '{:.4f}'.format(end_time - start_time))
        # ---------------------

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            final_boxes = self.post_processing(batch_dict)
            return final_boxes

    def get_training_loss(self):
        disp_dict = {}
        loss, tb_dict = self.reg_head.get_loss()

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        shape_x, shape_y, _ = batch_dict['motion_preds'][0].shape
        center_x, center_y = int(shape_x / 2), int(shape_y / 2)
        object_x = torch.round(center_x + batch_dict['motion_preds'][0, center_x, center_y, 0]).long()
        object_y = torch.round(center_y + batch_dict['motion_preds'][0, center_x, center_y, 1]).long()
        object_x = object_x if object_x < shape_x else shape_x - 1
        object_y = object_y if object_y < shape_y else shape_y - 1

        # ---------- ry prediction ----------
        ry_pred = batch_dict['ry_preds'][0, int(object_y / self.ration), int(object_x / self.ration)]  # Noted that position of object_x and object_y
        ry = torch.atan2(ry_pred[0], ry_pred[1])

        # ---------- z prediction ----------
        z = batch_dict['z_preds'][0, int(object_y / self.ration), int(object_x / self.ration)]  # Noted that position of object_x and object_y

        # ---------- motion prediction ----------
        assert self.voxel_size[0] == self.voxel_size[1]
        motion = batch_dict['motion_preds'][0, center_x, center_y] * self.motion_down_sample * self.voxel_size[0]

        # ---------- whl ----------
        whl = batch_dict['object_dim']  # B, 3

        final_box = torch.cat((motion.reshape(-1), z.reshape(-1), whl.reshape(-1), ry.reshape(-1)), dim=0)
        center_offset = batch_dict['center_offset'][0]
        final_box[:2] += center_offset[:2]

        ############################
        corner_off = batch_dict['predicted_box_corner_offset'][0, 0, :].reshape(3, 9).detach().cpu().numpy()
        center_off = np.mean(corner_off, axis=1)
        test_template_box = batch_dict['test_template_box'][0].cpu().numpy()
        motion_pred_box = copy.deepcopy(test_template_box)
        motion_pred_box[:2] = motion_pred_box[:2] + center_off[:2]

        return final_box, motion_pred_box

