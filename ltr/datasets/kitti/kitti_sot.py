import os
import copy
import pandas as pd
from pathlib import Path
from ..sotdataset import SOTDatasetTemplate
import numpy as np
import cv2
from .memory_bank import MemoryBank
from ...utils import box_utils, calibration_kitti, tracklet3d_kitti

class KittiSOTDataset(SOTDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, eval_flag=False, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, eval_flag=eval_flag, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.velodyne_path = os.path.join(self.root_path, 'velodyne')
        self.label_path = os.path.join(self.root_path, 'label_02')
        self.calib_path = os.path.join(self.root_path, 'calib')
        self.image_path = os.path.join(self.root_path, 'image_02')
        self.depth_image_path = os.path.join(self.root_path, 'depth_image_v3')
        self.history_frame_id_list = dataset_cfg.HISTORY_FRAME_IDS
        self.future_frame_id_list = dataset_cfg.FUTHURE_FRAME_IDS
        self.memory_bank = MemoryBank(abs(min(self.history_frame_id_list)))
        self.search_dim = [self.sr_point_cloud_range[x::3][1] - self.sr_point_cloud_range[x::3][0] for x in range(3)]
        self.motion_down_ratio = dataset_cfg.MOTION_DOWN_RATIO

        self.generate_split_list(self.split)

        self.refer_box = None
        self.first_points = None
        self.sequence_points = None
        self.template_area = None

    def generate_split_list(self, split):
        print('mode: ',self.mode)

        if split == 'train':
            self.sequence = list(range(0, 17))
        elif split == 'val':
            self.sequence = list(range(17, 19))
        elif split == 'test':
            self.sequence = list(range(19, 21))
        else:
            self.sequence = list(range(21))

        list_of_sequences = [
            path for path in os.listdir(self.velodyne_path)
            if os.path.isdir(os.path.join(self.velodyne_path, path)) and int(path) in self.sequence
        ]

        list_of_tracklet_anno = []
        self.first_frame_index = [0]
        number = 0
        for sequence in list_of_sequences:
            sequence_label_name = sequence + '.txt'
            label_file = os.path.join(self.label_path, sequence_label_name)

            seq_label = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame",
                    "track_id",
                    "type",
                    "truncated", "occlusion",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "ry"
                ])
            seq_label = seq_label[seq_label["type"] == self.class_names[0]]
            ########################
            # KITTI tracking dataset BUG
            if sequence == '0001':
                seq_label = seq_label[(~seq_label['frame'].isin([177,178,179,180]))]
            ########################

            seq_label.insert(loc=0, column="sequence", value=sequence)
            for track_id in seq_label.track_id.unique():
                seq_tracklet = seq_label[seq_label["track_id"] == track_id]
                seq_tracklet = seq_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in seq_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)
                number += len(tracklet_anno)
                self.first_frame_index.append(number)

        self.one_track_infos = self.get_whole_relative_frame(list_of_tracklet_anno)
        self.first_frame_index[-1] -= 1

    def get_whole_relative_frame(self, tracklets_infos):
        all_infos = []
        for one_infos in tracklets_infos:
            track_length = len(one_infos)
            relative_frame = 1
            for frame_info in one_infos:
                frame_info["relative_frame"] = relative_frame
                frame_info["track_length"] = track_length
                relative_frame += 1
                all_infos.append(frame_info)

        return all_infos

    def get_lidar(self, sequence, frame):
        lidar_file = os.path.join(self.velodyne_path, sequence, '{:06}.bin'.format(frame))
        assert Path(lidar_file).exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_label(self, idx):
        return self.one_track_infos[idx]

    def get_calib(self, sequence):
        calib_file = os.path.join(self.calib_path,'{}.txt'.format(sequence))
        assert Path(calib_file).exists()
        return calibration_kitti.Calibration(calib_file)

    def get_image(self, sequence, frame):
        image_file = os.path.join(self.image_path, sequence, '{:06}.png'.format(frame))
        assert Path(image_file).exists()
        return cv2.imread(image_file)

    def get_depth_image(self, sequence, frame):
        image_file = os.path.join(self.depth_image_path, sequence, '{:06}.npy'.format(frame))
        assert Path(image_file).exists()
        return np.load(image_file)[:, :, 0]

    def image_show_with_box(self, image, info):
        cv2.rectangle(image, (int(info['bbox_left']), int(info['bbox_top'])), (int(info['bbox_right']), int(info['bbox_bottom'])), (0, 0, 255), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def image_show_with_box_(self, image, label):
        cv2.rectangle(image, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), (0, 0, 255), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def image_show(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def rotat_point(self, points, ry):
        R_M = np.array([[np.cos(ry), -np.sin(ry), 0],
                                [np.sin(ry), np.cos(ry), 0],
                                [0, 0, 1]])  # (3, 3)
        rotated_point = np.matmul(points[:,:3], R_M)  # (N, 3)
        return rotated_point

    def __len__(self):
        return self.first_frame_index[-1] + 1

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'val':
            return self.get_train_tracking_item(index)
        else:
            return self.get_test_tracking_item(index)

    def find_random_template_idx(self, index, intervel=5):
        if self.mode == 'train' or self.mode == 'val':
            search_anno = self.one_track_infos[index]
            search_relative_frame = search_anno['relative_frame']
            search_whole_length = search_anno['track_length']
            search_min_index = max(0, search_relative_frame-intervel)
            search_max_index = min(search_relative_frame+intervel, search_whole_length)
            template_relative_frame = np.random.randint(search_min_index, search_max_index)
            template_index = index + template_relative_frame - search_relative_frame + 1
        else:
            template_index = index - 1

        return template_index

    def find_history_idx(self, index):
        pad_with_previous = self.history_frame_id_list
        template_list = [index + his_id for his_id in pad_with_previous]
        first_frame = np.array(self.first_frame_index)
        seq_index = first_frame[first_frame<=index][-1]
        template_list_re = [x if x >= seq_index else seq_index for x in template_list]
        return template_list_re

    def find_future_idx(self, index):
        pad_with_previous = self.future_frame_id_list
        template_list = [index + his_id for his_id in pad_with_previous]
        first_frame = np.array(self.first_frame_index)
        seq_index = first_frame[first_frame>index][0]
        template_list_re = [x if x < seq_index else seq_index-1 for x in template_list]
        return template_list_re

    # ================================== template point ==================================
    # ====================================================================================
    def template_point_zone(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        point_copy = copy.deepcopy(point)
        label = label.get_box3d()
        cx, cy, cz, dx, dy, dz, ry = label
        label_center = np.stack((cx, cy, cz))

        point_copy[:, :3] -= label_center
        norm_tpoints = point_copy

        # ------------ template_area ------------
        template_area = np.stack((0, 0, 0, sample_dx, sample_dy, sample_dz, 0))
        template_corner = box_utils.boxes_to_corners_3d(template_area.reshape(1, 7))[0]

        flag = box_utils.in_hull(norm_tpoints[:, :3], template_corner)
        template_area_points = norm_tpoints[flag]

        # ------------ template_box_area ------------
        template_box = np.stack((0, 0, 0, dx, dy, dz, ry))
        template_box_corner = box_utils.boxes_to_corners_3d(template_box.reshape(1, 7))[0]

        flag = box_utils.in_hull(template_area_points[:, :3], template_box_corner)
        template_box_points = template_area_points[flag]

        return template_area_points, template_box_points, template_box

    # ================================== search point ==================================
    # ==================================================================================
    def train_val_search_point_zone(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        point_copy = copy.deepcopy(point)
        label = label.get_box3d()
        cx, cy, cz, dx, dy, dz, ry = label
        label_center = np.stack((cx, cy, 0))

        point_copy[:, :3] -= label_center

        offset_ry = np.random.uniform(low=-0.4, high=0.4, size=1)[0]
        point_r = point_copy[:, 3].reshape(-1, 1)
        point_copy = self.rotat_point(point_copy, offset_ry)
        norm_tpoints = np.hstack((point_copy, point_r))

        offset_xy = np.random.uniform(low=-sample_dx / self.offset_down, high=sample_dx / self.offset_down, size=2)
        norm_tpoints[:, 0] = norm_tpoints[:, 0] - offset_xy[0]
        norm_tpoints[:, 1] = norm_tpoints[:, 1] - offset_xy[1]
        sample_center = np.stack((cx + offset_xy[0], cy + offset_xy[1], 0))

        search_area_z_center = (self.sr_point_cloud_range[2] + self.sr_point_cloud_range[-1]) / 2
        search_area = np.stack((0, 0, search_area_z_center, sample_dx, sample_dy, sample_dz, 0))
        # get points in search area
        search_corner = box_utils.boxes_to_corners_3d(search_area.reshape(1, 7))[0]

        flag = box_utils.in_hull(norm_tpoints[:, :3], search_corner)
        crop_points = norm_tpoints[flag]

        new_box = np.stack(
            (-offset_xy[0], -offset_xy[1], cz, dx, dy, dz, ry-offset_ry))
        new_box_corner = box_utils.boxes_to_corners_3d(new_box.reshape(1, 7))[0].transpose()

        return offset_ry, crop_points, sample_center, new_box, search_corner, search_area

    def test_search_point_zone(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        label = label.get_box3d()
        cx, cy, cz, dx, dy, dz, ry = label
        sample_center = np.stack((cx, cy, 0))

        search_area_z_center = (self.sr_point_cloud_range[2] + self.sr_point_cloud_range[-1]) / 2
        area = np.stack((cx, cy, search_area_z_center, sample_dx, sample_dy, sample_dz, 0)).reshape(1, 7)
        corner = box_utils.boxes_to_corners_3d(area)[0]
        object_flag = box_utils.in_hull(point[:, :3], corner)
        area_points = point[object_flag]
        area_points[:, :3] -= sample_center

        new_box = np.stack(
            (cx, cy, cz, dx, dy, dz, ry))
        new_box_corner = box_utils.boxes_to_corners_3d(new_box.reshape(1, 7))[0].transpose()

        return area_points, sample_center, new_box, corner, area

    # ================================== bev mask ==================================
    def generate_template_box_mask(self, template_box):
        shape = (self.sr_grid_size[:2] / 8).astype(int)
        gt_map = np.zeros((shape))
        gt_mask = box_utils.get_bev_box_mask(template_box, shape[0], shape[1], self.sr_point_cloud_range,
                                             [x * 8 for x in self.voxel_size])

        gt_map = gt_map.reshape(-1, 1)
        gt_map[gt_mask] = 1
        template_box_mask = gt_map.reshape(*shape, order='F')

        return template_box_mask

    def generate_ry_mask(self, template_box, local_search_new_box):
        shape = (self.sr_grid_size[:2] / self.motion_down_ratio).astype(int)
        gt_map = np.zeros((shape))
        gt_mask = box_utils.get_bev_box_mask(template_box, shape[0], shape[1], self.sr_point_cloud_range,
                                             [x * self.motion_down_ratio for x in self.voxel_size])

        gt_map = gt_map.reshape(-1, 1)
        gt_map[gt_mask] = local_search_new_box[-1] - template_box[-1]
        template_box_mask = gt_map.reshape(*shape, order='F')

        return template_box_mask

    def generate_motion_mask(self, template_box):
        shape = (self.sr_grid_size[:2] / self.motion_down_ratio).astype(int)
        gt_map = np.zeros((shape))
        gt_mask = box_utils.get_bev_box_mask(template_box, shape[0], shape[1], self.sr_point_cloud_range,
                                             [x * self.motion_down_ratio for x in self.voxel_size])

        gt_map = gt_map.reshape(-1, 1)
        gt_map[gt_mask] = 1
        template_box_mask = gt_map.reshape(*shape, order='F')

        return template_box_mask

    def generate_motion(self, template_box, local_search_new_box):
        off_xy = (local_search_new_box[:2] / self.voxel_size[:2] / self.motion_down_ratio).reshape(1, -1)
        off_ry = local_search_new_box[-1] - template_box[-1]
        cosa = np.cos(off_ry)
        sina = np.sin(off_ry)
        rot_matrix = np.array((
            cosa, sina,
            -sina, cosa,
        )).reshape(2, 2)

        shape = (self.sr_grid_size[:2] / self.motion_down_ratio).astype(int)
        ct2d = (shape / 2).astype(int).reshape(1, 2)
        gt_mask = box_utils.get_bev_box_mask(template_box, shape[0], shape[1], self.sr_point_cloud_range,
                                             [x * self.motion_down_ratio for x in self.voxel_size])

        grid_x, grid_y = np.meshgrid(np.linspace(0, shape[1] - 1, shape[1]),
                                     np.linspace(0, shape[0] - 1, shape[0]))
        img_grid = np.stack([grid_x, grid_y], -1).reshape(-1, 2)
        template_box_grid = img_grid[gt_mask]

        search_box_grid = np.matmul((template_box_grid - ct2d), rot_matrix) + ct2d
        search_box_grid = search_box_grid + off_xy
        search_box_grid = np.around(search_box_grid).astype('int')
        template_box_grid = template_box_grid.astype('int')

        motion_map = np.zeros((shape[0], shape[1], 2))
        box_motion = search_box_grid - template_box_grid
        motion_map[template_box_grid[:, 0], template_box_grid[:, 1], 0] = box_motion[:, 0]
        motion_map[template_box_grid[:, 0], template_box_grid[:, 1], 1] = box_motion[:, 1]
        motion_map[ct2d[0][0], ct2d[0][1], 0], motion_map[ct2d[0][0], ct2d[0][1], 1] = off_xy[0, 0], off_xy[0, 1]

        return motion_map

    # ================================== generate history box ==================================
    def train_val_generate_history_box(self, search_box, history_box_list, local_search_box, local_template_box):
        search_box_ = copy.deepcopy(search_box)
        local_search_box_ = copy.deepcopy(local_search_box)
        local_template_box_ = copy.deepcopy(local_template_box)
        history_box_num = history_box_list.shape[0]

        ################# motion xy
        history_box = np.zeros((history_box_num - 1, 7))
        origin_motion_list = search_box_ - history_box_list
        current_search_motion = local_search_box_[:2]
        ################# motion ry
        current_search_ry_motion = local_search_box_[-1] - local_template_box_[-1]

        for ind in range(history_box_num - 1):
            #################### object xy
            rate = origin_motion_list[0][:2] / (origin_motion_list[ind + 1][:2] + 1e-8)
            all_length = current_search_motion / (rate + 1e-8)
            object_length = all_length - current_search_motion
            #################### object ry
            rate = origin_motion_list[0][-1] / (origin_motion_list[ind + 1][-1] + 1e-8)
            object_ry = current_search_ry_motion / (rate + 1e-8)
            object_ry = local_search_box_[-1] - object_ry

            history_box[ind, :2] = -object_length
            history_box[ind, 3:-1] = search_box_[3:-1]
            history_box[ind, -1] = object_ry

        return history_box

    def test_generate_history_box(self, history_box_list, refer_box):
        if self.memory_bank.cur_length() == 0:
            cx, cy = refer_box.get_box3d()[:2]
            refer_boxes = history_box_list
        else:
            memory_boxes = self.memory_bank.return_box()
            pad_box = np.repeat(memory_boxes[-1:], self.memory_bank.max_length - self.memory_bank.cur_length(), axis=0)
            memory_boxes = np.vstack((memory_boxes, pad_box))

            refer_id_list = [-x - 1 for x in self.history_frame_id_list]
            refer_boxes = memory_boxes[refer_id_list]
            cx, cy = refer_boxes[0][:2]

        local_template_gtbox_list = []
        for k in range(len(history_box_list)):
            box = refer_boxes[k]
            box[0] -= cx
            box[1] -= cy
            box[2] = 0
            local_template_gtbox_list.append(box)
        return local_template_gtbox_list

    # ================================== generate history box ==================================
    def train_val_generate_future_box(self, search_box, history_box_list, local_template_box, future_box_list, local_search_box, random_ry):
        search_box_ = copy.deepcopy(search_box)
        local_search_box_ = copy.deepcopy(local_search_box)
        template_box_ = copy.deepcopy(history_box_list[0])
        future_box_num = future_box_list.shape[0]

        ################# motion xy
        future_box = np.zeros((future_box_num, 7))
        origin_search_motion = search_box_ - template_box_
        origin_future_motion_list = future_box_list - template_box_
        ################# motion xy
        current_search_motion = local_search_box_ - local_template_box

        for ind in range(future_box_num):
            #################### object xy
            rate = origin_search_motion[:2] / (local_search_box_[:2] + 1e-8)
            object_length = origin_future_motion_list[ind][:2] / (rate + 1e-8)
            #################### object ry
            rate = origin_search_motion[-1] / (current_search_motion[-1] + 1e-8)
            object_ry = origin_future_motion_list[ind][-1] / (rate + 1e-8) + local_template_box[-1]

            future_box[ind, :2] = object_length
            future_box[ind, 3:-1] = search_box_[3:-1]
            future_box[ind, -1] = object_ry

        return future_box

    def get_train_tracking_item(self, index):
        # ---------- search frame ----------
        search_info = self.get_label(index)
        gt_name = search_info['type']  # search -> type
        search_sequence = search_info['sequence']  # search -> sequence
        search_frame = search_info['frame']  # search -> frame
        search_calib = self.get_calib(search_sequence)  # search -> calib
        search_points = self.get_lidar(search_sequence, search_frame)  # search -> points

        search_boxes = tracklet3d_kitti.Tracklet3d_Camera(search_info)
        search_boxes_lidar = search_boxes.get_lidar_box3d(search_calib).reshape(-1)   # search -> gt_box (lidar)
        search_box = tracklet3d_kitti.Tracklet3d_Lidar(search_boxes_lidar)

        # ---------- template frame----------
        template_index = self.find_random_template_idx(index)
        template_info = self.get_label(template_index)  # template -> label
        template_sequence = template_info['sequence']  # template -> sequence
        template_frame = template_info['frame']  # template -> frame
        template_calib = self.get_calib(template_sequence)  # template -> calib
        template_points = self.get_lidar(template_sequence, template_frame)  # template -> points

        template_gt_box = tracklet3d_kitti.Tracklet3d_Camera(template_info)
        template_gt_box_lidar = template_gt_box.get_lidar_box3d(template_calib).reshape(-1)   # search -> gt_box (lidar)
        template_box = tracklet3d_kitti.Tracklet3d_Lidar(template_gt_box_lidar)

        # ---------- search & template point area ----------
        random_ry, local_search_points, center_offset, local_search_new_box, local_search_corner, local_search_area, \
        local_template_points, local_template_box_points, local_template_box = self.train_val_point_zone_func(search_box,
                                                                                                              template_box,
                                                                                                              search_points,
                                                                                                              template_points)
        if local_template_points.shape[0] <= 20 or local_search_points.shape[0] <= 20:  # points
            return self.get_train_tracking_item(np.random.randint(0, self.__len__()))

        # ---------- template bev mask ----------
        template_bev_mask = self.generate_template_box_mask(local_template_box)
        motion_mask = self.generate_motion_mask(local_template_box)
        motion_map = self.generate_motion(local_template_box, local_search_new_box)
        ry_map = self.generate_ry_mask(local_template_box, local_search_new_box)

        # ---------- history frame ----------
        history_index_list = self.find_history_idx(index)
        history_index_state = [history_index == index for history_index in history_index_list]
        if not False in history_index_state:
            return self.get_train_tracking_item(np.random.randint(0, self.__len__()))
        history_info_list = [self.get_label(tindex) for tindex in history_index_list]  # template -> label
        history_gt_box_list = list(map(tracklet3d_kitti.Tracklet3d_Camera, history_info_list))
        history_gt_box_lidar_list = np.vstack(list([one_box.get_lidar_box3d(search_calib).reshape(-1) for one_box in history_gt_box_list]))  # search -> gt_box (lidar)

        history_box = self.train_val_generate_history_box(search_boxes_lidar, history_gt_box_lidar_list, local_search_new_box, local_template_box)
        local_search_new_box[2] = 0

        history_box_corner = list([tracklet3d_kitti.Tracklet3d_Lidar(one_box).corners() for one_box in history_box])
        history_box_center = list([one_box[:3].reshape(-1, 1) for one_box in history_box])
        history_box_ry = list([one_box[-1].reshape(-1, 1) for one_box in history_box])
        for ind in range(history_box.shape[0]):
            history_box_corner[ind] = np.concatenate((history_box_corner[ind], history_box_center[ind]), axis=-1).reshape(1, -1)
            history_box_corner[ind] = np.concatenate((history_box_corner[ind], history_box_ry[ind]), axis=-1)

        local_template_box_corner = tracklet3d_kitti.Tracklet3d_Lidar(local_template_box).corners()
        local_template_box_corner = np.concatenate((local_template_box_corner, local_template_box[:3].reshape(-1, 1)), axis=-1)
        history_box_corner.insert(0, np.concatenate((local_template_box_corner.reshape(1, -1), local_template_box[-1].reshape(-1, 1)), axis=-1))
        history_box_corner = np.vstack(history_box_corner)

        local_search_box_corner = tracklet3d_kitti.Tracklet3d_Lidar(local_search_new_box).corners()
        local_search_box_corner = np.concatenate((local_search_box_corner, local_search_new_box[:3].reshape(-1, 1)), axis=-1)

        gt_box_corner_offset = local_search_box_corner - local_template_box_corner
        gt_box_ry_offset = local_search_new_box[-1] - local_template_box[-1]

        search_boxes_lidar[-1] = local_search_new_box[-1]
        input_dict = {
            'gt_names': gt_name,
            'gt_boxes': search_boxes_lidar,
            'search_points': local_search_points,
            'center_offset': center_offset.reshape(1, -1),

            'object_dim': local_template_box[3:6].reshape(1, 3),
            'template_box': local_template_box.reshape(1, 7),
            'template_gt_box': template_gt_box_lidar.reshape(1, 7),
            'template_points': local_template_points,
            'template_bev_mask': template_bev_mask,
            'motion_mask': motion_mask,

            'motion_map': motion_map,
            'ry_map': ry_map,
            'history_box_corner': history_box_corner,
            'gt_box_corner_offset': gt_box_corner_offset.reshape(1, -1),
            'gt_box_ry_offset': gt_box_ry_offset.reshape(1, -1)
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.mode == 'train' or self.mode == 'val':
            tv = data_dict['template_voxels']
            sv = data_dict['search_voxels']
            if sv.shape[0] <= 20 or tv.shape[0] <= 20:
                return self.get_train_tracking_item(np.random.randint(0, self.__len__()))

        return data_dict

    def get_test_tracking_item(self, index):
        # ---------- search frame ----------
        search_info = self.get_label(index)
        gt_name = search_info['type']  # search -> type
        search_sequence = search_info['sequence']  # search -> sequence
        search_frame = search_info['frame']  # search -> frame
        search_calib = self.get_calib(search_sequence)  # search -> calib
        search_points = self.get_lidar(search_sequence, search_frame)  # search -> points

        search_boxes = tracklet3d_kitti.Tracklet3d_Camera(search_info)
        search_boxes_lidar = search_boxes.get_lidar_box3d(search_calib).reshape(-1)   # search -> gt_box (lidar)
        search_box = tracklet3d_kitti.Tracklet3d_Lidar(search_boxes_lidar)

        # ---------- template frame----------
        template_index = self.find_random_template_idx(index)
        template_info = self.get_label(template_index)  # template -> label
        template_sequence = template_info['sequence']  # template -> sequence
        template_frame = template_info['frame']  # template -> frame
        template_calib = self.get_calib(template_sequence)  # template -> calib
        template_points = self.get_lidar(template_sequence, template_frame)  # template -> points

        template_gt_box = tracklet3d_kitti.Tracklet3d_Camera(template_info)
        template_gt_box_lidar = template_gt_box.get_lidar_box3d(template_calib).reshape(-1)   # search -> gt_box (lidar)
        template_box = tracklet3d_kitti.Tracklet3d_Lidar(template_gt_box_lidar)

        # ---------- search & template point area ----------
        local_search_points, center_offset, local_search_new_box, local_search_corner, search_area, \
        local_template_points, local_template_box_points, local_template_box = self.test_point_zone_func(search_box,
                                                                                                         template_box,
                                                                                                         search_points,
                                                                                                         template_points)

        # ---------- bev mask ----------
        template_bev_mask = self.generate_template_box_mask(local_template_box)

        # ---------- history frame ----------
        history_index_list = self.find_history_idx(index)
        history_info_list = [self.get_label(tindex) for tindex in history_index_list]  # template -> label
        history_gt_box_list = list(map(tracklet3d_kitti.Tracklet3d_Camera, history_info_list))
        history_gt_box_lidar_list = np.vstack(list([one_box.get_lidar_box3d(search_calib).reshape(-1) for one_box in history_gt_box_list]))  # search -> gt_box (lidar)

        history_box = self.test_generate_history_box(history_gt_box_lidar_list, template_box)
        history_box_corner = list([tracklet3d_kitti.Tracklet3d_Lidar(one_box).corners() for one_box in history_box])
        history_box_center = list([one_box[:3].reshape(-1, 1) for one_box in history_box])
        history_box_ry = list([one_box[-1].reshape(-1, 1) for one_box in history_box])
        for ind in range(len(history_box)):
            history_box_corner[ind] = np.concatenate((history_box_corner[ind], history_box_center[ind]), axis=-1).reshape(1, -1)
            history_box_corner[ind] = np.concatenate((history_box_corner[ind], history_box_ry[ind]), axis=-1)
        history_box_corner = np.vstack(history_box_corner)

        if self.memory_bank.cur_length() == 0:
            test_template_box = template_gt_box_lidar
        else:
            test_template_box = self.memory_bank.return_box()[0]

        input_dict = {
            'gt_names': gt_name,
            'gt_boxes': search_boxes_lidar,
            'search_points': local_search_points,
            'center_offset': center_offset.reshape(1, -1),

            'object_dim': local_template_box[3:6].reshape(1, 3),
            'template_box': local_template_box.reshape(1, 7),
            'template_gt_box': template_gt_box_lidar.reshape(1, 7),
            'template_bev_mask': template_bev_mask,

            'history_box_corner': history_box_corner,
            'test_template_box': test_template_box,
        }

        input_dict.update({
            'or_search_points': local_search_points,
            'or_local_search_corner': local_search_corner,
            'or_template_points': local_template_points,
            'or_template_box_points': local_template_box_points,
            'ori_search_points': search_points,
        })

        input_dict.update({
            'template_points': local_template_points,
        })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    # ================== point zone ==================
    def train_val_point_zone_func(self, gt_boxes_lidar, template_gt_box_lidar, search_points, template_points):
        random_ry, local_search_points, center_offset, local_search_new_box, \
        local_search_corner, local_search_area = self.train_val_search_point_zone(search_points,
                                                                            gt_boxes_lidar,
                                                                            self.search_dim[0],
                                                                            self.search_dim[1],
                                                                            self.search_dim[2])
        local_template_points, \
        local_template_box_points, local_template_box = self.template_point_zone(template_points,
                                                                     template_gt_box_lidar,
                                                                     self.search_dim[0],
                                                                     self.search_dim[1],
                                                                     self.search_dim[2])

        return random_ry, local_search_points, center_offset, local_search_new_box, local_search_corner, local_search_area,\
               local_template_points, local_template_box_points, local_template_box

    def test_point_zone_func(self, gt_boxes_lidar, template_gt_box_lidar, search_points, template_points):
        if self.memory_bank.cur_length() == 0:
            search_box = template_gt_box_lidar
            template_box = template_gt_box_lidar
        else:
            search_box = tracklet3d_kitti.Tracklet3d_Lidar(self.memory_bank.return_box()[0])
            template_box = tracklet3d_kitti.Tracklet3d_Lidar(self.memory_bank.return_box()[0])

        local_search_points, center_offset, local_search_new_box, \
        local_search_corner, search_area = self.test_search_point_zone(search_points, search_box,
                                                                       self.search_dim[0], self.search_dim[1],
                                                                       self.search_dim[2])
        template_area_points, \
        template_box_points, template_box = self.template_point_zone(template_points, template_box,
                                                                     self.search_dim[0],
                                                                     self.search_dim[1],
                                                                     self.search_dim[2])

        return local_search_points, center_offset, local_search_new_box, local_search_corner, search_area, \
               template_area_points, template_box_points, template_box


    def set_refer_box(self, refer_box):
        self.refer_box = refer_box

    def set_first_points(self, points):
        self.first_points = points

    def add_refer_box(self, refer_box):
        self.memory_bank.add(refer_box)

    def reset_all(self):
        self.memory_bank.reset()