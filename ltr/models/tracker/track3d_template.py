import os
import torch
import torch.nn as nn
from .. import backbones_2d, backbones_3d, neck, reg_head, motion_predictor
from ..backbones_2d import map_to_bev
from ..backbones_3d import vfe

class Track3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'motion_predictor', 'vfe', 'backbone_3d', 'map_to_bev_module',
            'backbone_2d', 'neck', 'reg_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'sr_grid_size': self.dataset.sr_grid_size,
            'tp_grid_size': self.dataset.tp_grid_size,
            'sr_point_cloud_range': self.dataset.sr_point_cloud_range,
            'tp_point_cloud_range': self.dataset.tp_point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_motion_predictor(self, model_info_dict):
        if self.model_cfg.get('MOTION_PREDICTOR', None) is None:
            return None, model_info_dict

        motion_predictor_module = motion_predictor.__all__[self.model_cfg.MOTION_PREDICTOR.NAME](
            model_cfg=self.model_cfg.MOTION_PREDICTOR
        )
        model_info_dict['module_list'].append(motion_predictor_module)
        return motion_predictor_module, model_info_dict

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            sr_grid_size=model_info_dict['sr_grid_size'],
            tp_grid_size=model_info_dict['tp_grid_size'],
            sr_point_cloud_range=model_info_dict['sr_point_cloud_range'],
            tp_point_cloud_range=model_info_dict['tp_point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            sr_grid_size=model_info_dict['sr_grid_size'],
            tp_grid_size=model_info_dict['tp_grid_size'],
            sr_point_cloud_range=model_info_dict['sr_point_cloud_range'],
            tp_point_cloud_range=model_info_dict['tp_point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            sr_grid_size=model_info_dict['sr_grid_size'],
            tp_grid_size=model_info_dict['tp_grid_size'],
            sr_point_cloud_range=model_info_dict['sr_point_cloud_range'],
            tp_point_cloud_range=model_info_dict['tp_point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['features_lists'] = backbone_2d_module.feats_list
        return backbone_2d_module, model_info_dict

    def build_neck(self, model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict

        neck_module = neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK,
            # input_channels_lists=model_info_dict['features_lists']
            input_channels_lists = model_info_dict['features_lists'] if model_info_dict.get('features_lists', None) else None,
            sr_point_cloud_range=model_info_dict['sr_point_cloud_range'],
            tp_point_cloud_range=model_info_dict['tp_point_cloud_range'],
            voxel_size = model_info_dict['voxel_size']
        )
        model_info_dict['module_list'].append(neck_module)
        model_info_dict['features_lists'] = neck_module.feats_list
        model_info_dict['num_neck_features'] = neck_module.num_neck_features
        return neck_module, model_info_dict
    
    def build_reg_head(self, model_info_dict):
        if self.model_cfg.get('REG_HEAD', None) is None:
            return None, model_info_dict
        reg_head_module = reg_head.__all__[self.model_cfg.REG_HEAD.NAME](
            model_cfg=self.model_cfg.REG_HEAD,
            input_channels=model_info_dict['num_neck_features'],
            grid_size=model_info_dict['sr_grid_size'],
            point_cloud_range=model_info_dict['sr_point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
        )
        model_info_dict['module_list'].append(reg_head_module)
        return reg_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        final_box = batch_dict['predict_box']

        return final_box

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
