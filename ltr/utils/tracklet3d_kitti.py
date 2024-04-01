import numpy as np
from . import common_utils
from pyquaternion import Quaternion

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Tracklet3d_Camera(object):
    def __init__(self, tracklet):
        self.frome = tracklet['frame']
        self.cls_type = tracklet['type']
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(tracklet['truncated'])
        self.occlusion = float(tracklet['occlusion'])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(tracklet['alpha'])
        self.box2d = np.array((float(tracklet['bbox_left']), float(tracklet['bbox_top']), float(tracklet['bbox_right']), float(tracklet['bbox_bottom'])), dtype=np.float32)
        self.h = float(tracklet['height'])
        self.w = float(tracklet['width'])
        self.l = float(tracklet['length'])
        self.center = np.array((float(tracklet['x']), float(tracklet['y']), float(tracklet['z']), ), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.center)
        self.ry = float(tracklet['ry'])
        self.score = float(tracklet['score']) if tracklet.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

        self.lidar_orientation = Quaternion(
                axis=[0, 1, 0], radians=self.ry) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.center, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.center[0], self.center[1], self.center[2],
                       self.ry)
        return kitti_str

    # camera
    def get_camera_box3d(self):
        boxes3d = np.stack((self.center[0], self.center[1], self.center[2], self.l, self.h, self.w, self.ry))
        return boxes3d

    # lidar
    def get_lidar_box3d(self, calib):
        camera_boxes3d = self.get_camera_box3d().reshape(-1, 7)

        xyz_camera = camera_boxes3d[:, 0:3]
        l, h, w, r = camera_boxes3d[:, 3:4], camera_boxes3d[:, 4:5], camera_boxes3d[:, 5:6], camera_boxes3d[:, 6:7]
        xyz_lidar = calib.rect_to_lidar(xyz_camera)
        xyz_lidar[:, 2] += h[:, 0] / 2
        return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

    # camera
    def generate_camera_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.center
        return corners3d


class Tracklet3d_Lidar(object):
    def __init__(self, anno_lidar):
        self.center = anno_lidar[:3]
        self.dim = anno_lidar[3:6]
        self.ry = np.array(anno_lidar[6]).reshape(-1)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=anno_lidar[6])

    def get_box3d(self):
        boxes3d = np.concatenate((self.center, self.dim, self.ry))
        return boxes3d

    def translate(self, x):
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        :return: <None>.
        """
        self.center += x

    def rotate(self, quaternion):
        """
        Rotates box.
        :param quaternion: <Quaternion>. Rotation to apply.
        :return: <None>.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation

    @property
    def rotation_matrix(self):
        """
        Return a rotation matrix.
        :return: <np.float: (3, 3)>.
        """
        return self.orientation.rotation_matrix

    def corners(self, wlh_factor=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        dx = self.dim[0] * wlh_factor
        dy = self.dim[1] * wlh_factor
        dz = self.dim[2] * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = dx / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = dy / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = dz / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

class PointCloud:
    def __init__(self, points):
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: 4, n>. Input point cloud matrix.
        """
        self.points = points
        if self.points.shape[0] > 3:
            self.points = self.points[0:3, :]

    def nbr_points(self):
        """
        Returns the number of points.
        :return: <int>. Number of points.
        """
        return self.points.shape[1]

    def translate(self, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        :return: <None>.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        :return: <None>.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix):
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        :return: <None>.
        """
        self.points[:3, :] = transf_matrix.dot(
            np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]