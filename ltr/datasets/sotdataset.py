from collections import defaultdict
from pathlib import Path
import numpy as np
import torch.utils.data as torch_data
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
import cv2

class Image_List(object):
    def __init__(self, image_list):
        self.image_list = image_list

    def _max_by_axis(self, image_size):
        maxes = image_size[0]
        for sublist in image_size[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def nested_from_list(self):
        max_size = self._max_by_axis([list(img.shape) for img in self.image_list])
        batch_shape = [len(self.image_list)] + max_size
        b, c, h, w = batch_shape
        dtype = self.image_list[0].dtype
        numpy = np.zeros(batch_shape, dtype=dtype)
        mask = np.ones((b, h, w), dtype=np.bool)
        for img, pad_img, m in zip(self.image_list, numpy, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
            m[: img.shape[1], :img.shape[2]] = False

        return numpy, mask

class SOTDatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, eval_flag=False, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.eval_flag = eval_flag
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        # self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.sr_point_cloud_range = np.array(self.dataset_cfg.SR_POINT_CLOUD_RANGE, dtype=np.float32)
        self.tp_point_cloud_range = np.array(self.dataset_cfg.TP_POINT_CLOUD_RANGE, dtype=np.float32)

        self.offset_down = self.dataset_cfg.get("OFFSET_XY_DOWN", 4)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING)

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, sr_point_cloud_range=self.sr_point_cloud_range, tp_point_cloud_range=self.tp_point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.sr_grid_size = self.data_processor.sr_grid_size
        self.tp_grid_size = self.data_processor.tp_grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        if not self.eval_flag:
            return 'train' if self.training else 'val'
        else:
            return 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if data_dict.get('gt_boxes', None) is not None:
            gt_classes = np.array([self.class_names.index(data_dict['gt_names']) + 1], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'].reshape(1,7), gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['search_voxels', 'search_voxel_num_points', 'template_voxels', 'template_voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['object_dim', 'center_offset', 'center_ry']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['or_search_points','or_template_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['search_points', 'template_points', 'search_voxel_coords', 'template_voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['refer_box']:
                    ret[key] = np.stack(val, axis=0)
                elif key in ['template_bev_mask', 'motion_mask']:
                    ret[key] = np.stack(val, axis=0)
                elif key in ['motion_map', 'ry_map', 'z_map']:
                    ret[key] = np.stack(val, axis=0)
                elif key in ['gt_info']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
