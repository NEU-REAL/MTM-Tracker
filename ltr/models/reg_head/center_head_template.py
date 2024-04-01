import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import loss_utils
from torch.autograd import Variable
from ltr.ops.points_op import points_op_cpu
from ltr.utils.loss_utils import weighted_smoothl1, weighted_sigmoid_focal_loss

class MultiCEFocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()

        self.alpha = Variable(torch.ones(class_num, 1))
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        self.alpha = self.alpha.to(predict.device)
        pt = predict.permute(1, 0)
        class_mask = F.one_hot(target, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = torch.log(probs + 1e-5)

        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class CenterHeadTemplate(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, voxel_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels

        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_num = self.model_cfg.MAX_NUM
        self.assign_radius = self.model_cfg.ASSIGN_RADIUS
        if model_cfg.LOSS_CONFIG.LOSS_MT == 'L1':
            self.mt_loss_func = torch.nn.L1Loss(reduction='none')
        elif model_cfg.LOSS_CONFIG.LOSS_MT == 'SmoothL1Loss':
            self.mt_loss_func = torch.nn.SmoothL1Loss(reduction='none')
        elif model_cfg.LOSS_CONFIG.LOSS_MT == 'Huber Loss':
            self.mt_loss_func = torch.nn.HuberLoss(reduction='none')
        else:
            raise
        self.cls_loss_func = loss_utils.FocalLoss()
        self.seg_loss_func = nn.BCELoss()
        self.forward_ret_dict = {}

    def make_final_layers(self, input_channel, layer_list):
        layers = []
        pre_channel = input_channel
        for k in range(0, len(layer_list)-1):
            layers.extend([
                nn.Conv2d(pre_channel, layer_list[k], kernel_size=1, bias=False),
                nn.BatchNorm2d(layer_list[k]),
                nn.ReLU()
            ])
            pre_channel = layer_list[k]
        layers.append(nn.Conv2d(pre_channel, layer_list[-1], kernel_size=1, bias=True))
        layers = nn.Sequential(*layers)
        return layers

    def transformer2bev(self, points):
        dtype = points.dtype
        device = points.device

        pc_range = torch.tensor(self.pc_range, dtype=dtype, device=device)
        voxel_size = torch.tensor(self.voxel_size, dtype=dtype, device=device)

        points2d = ((points[:,:2]-pc_range[:2])/voxel_size[:2])/self.feature_map_stride
        return points2d

    def filter_point(self, points, height, width):
        points = torch.unique(points,dim=0)
        keep_x = ((points[:,0] >= 0) & (points[:,0] < width))
        points = points[keep_x]
        keep_y = ((points[:,1] >= 0) & (points[:,1] < height))
        points = points[keep_y]
        return points

    def make_fake_spots(self, hotspots2d, height, width):
        '''
        '''
        left = hotspots2d.clone()
        right = hotspots2d.clone()
        top = hotspots2d.clone()
        bottom = hotspots2d.clone()
        left[:,0] += 1
        right[:,0] -= 1
        top[:,1] += 1
        bottom[:,1] -= 1
        fake = torch.cat((left, right, top, bottom), dim=0)
        
        fake = self.filter_point(fake, height, width)
        return fake

    def get2d_fg(self, gt_box, height, width):
        dtype = gt_box.dtype
        device = gt_box.device
        gt_box = gt_box.squeeze(0)

        pc_range = torch.tensor(self.pc_range, dtype=dtype, device=device)
        voxel_size = torch.tensor(self.voxel_size, dtype=dtype, device=device)

        # -----ry-----
        ry = gt_box[6].view(1)
        cosa = torch.cos(ry)
        sina = torch.sin(ry)
        rot_matrix = torch.stack((
            cosa,  sina,
            -sina, cosa, 
        ), dim=1).view(2,2).float()

        # -----center-----
        ct2d = ((gt_box[:2]-pc_range[:2])/voxel_size[:2])/self.feature_map_stride
        ct2d = ct2d.long()

        # -----wl-----
        w2d = ((gt_box[3:4]/voxel_size[0])/(self.feature_map_stride*2)).long().item()
        l2d = ((gt_box[4:5]/voxel_size[1])/(self.feature_map_stride*2)).long().item()

        # -----gt Feature pixel-----
        fg2d = []
        for x in range(-w2d-1, w2d+2):
            for y in range(-l2d-1, l2d+2):
                pixel = torch.tensor([x,y], device=device).view(1,2).long()
                fg2d.append(pixel)
        fg2d = torch.cat((fg2d),dim=0)
        fg2d_rot = torch.matmul(fg2d[:, 0:2].float(), rot_matrix).long()
        fg2d_rot[:,:2] += ct2d[:2]

        # -----filter point-----
        fg2d_rot = self.filter_point(fg2d_rot, height, width)
        return fg2d_rot

    def assigne_hot(self, gt_box, feature_map_size, max_num, height, template_point):
        '''
        gt_box: 1, 7
        AFDet-version, assign only ct point for regressioin
        '''
        dtype = gt_box.dtype
        device = gt_box.device

        heatmap = gt_box.new_zeros(*feature_map_size)
        regmap = gt_box.new_zeros((max_num, 8))
        maskmap = gt_box.new_zeros((max_num)).long()
        indmap = gt_box.new_zeros((max_num)).long()

        feat_h, feat_w = map(int, feature_map_size)

        fg2d = self.get2d_fg(gt_box, feat_h, feat_w)

        center_2d = self.transformer2bev(gt_box)
        center_2d_int = center_2d.long()

        depth2dc = torch.norm(fg2d.float()-center_2d_int.float(), dim=1, p=2, keepdim=False)
        heatmap[fg2d[:,1], fg2d[:,0]] = torch.reciprocal(depth2dc)

        fake = self.make_fake_spots(center_2d_int, feat_h, feat_w)
       
        heatmap[fake[:,1],fake[:,0]] = 0.8
        heatmap[center_2d_int[:,1],center_2d_int[:,0]] = 1

        hotspots2d = []
        ctx = center_2d_int[:,0].item()
        cty = center_2d_int[:,1].item()
        for x in range(max(0, ctx-self.assign_radius), min(ctx+self.assign_radius+1, feat_w)):
            for y in range(max(0, cty-self.assign_radius), min(cty+self.assign_radius+1, feat_h)):
                pixel = torch.tensor((x,y), device=device).view(1,2)
                hotspots2d.append(pixel)

        hotspots2d = torch.cat(hotspots2d)

        hotspots2d = self.filter_point(hotspots2d, feat_h, feat_w)

        ind = hotspots2d[:,1] * feat_h + hotspots2d[:,0]
        object_num = hotspots2d.shape[0]
        
        indmap[:object_num] = ind

        center_res2d = center_2d - hotspots2d[:,:2].float()
        regmap[:object_num, :2] = center_res2d
        regmap[:object_num, 2] = height
        regmap[:object_num, 3:6] = gt_box.view(-1)[3:6]
        regmap[:object_num, 6] = torch.sin(gt_box.view(-1)[6])
        regmap[:object_num, 7] = torch.cos(gt_box.view(-1)[6])

        maskmap[:object_num] = 1

        return heatmap, regmap, maskmap, indmap
    
    def assign_stack_targets(self, gt_boxes, real_height, search_points):
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        
        feature_map_size = (self.grid_size[:2] // self.feature_map_stride)[[1,0]]
        
        batch_size = gt_boxes.shape[0]

        cls_labels = gt_boxes.new_zeros(batch_size, *feature_map_size)  # [batch_size, feature_map_size, feature_map_size]
        reg_labels = gt_boxes.new_zeros((batch_size, self.max_num, 8))  # [batch_size, max_num, 5]
        reg_mask = gt_boxes.new_zeros((batch_size, self.max_num)).long()  # [batch_size, max_num]
        ind_labels = gt_boxes.new_zeros((batch_size, self.max_num)).long()  # [batch_size, max_num]

        for k in range(batch_size):
            search_point = search_points[(search_points[:, 0] == k)][:, 1:]
            gt_box = gt_boxes[k]  # 1, 8
            height = real_height[k].view(-1)
            heatmap, regmap, maskmap, indmap = self.assigne_hot(gt_box[:,:7], feature_map_size, self.max_num, height, search_point)
            cls_labels[k] = heatmap
            reg_labels[k] = regmap
            ind_labels[k] = indmap
            reg_mask[k] = maskmap

        targets_dict = {
            'cls_labels': cls_labels,
            'reg_labels': reg_labels,
            'reg_mask': reg_mask,
            'ind_labels':ind_labels,
        }
        return targets_dict

    def _gather_feat(self, feat, index, mask=None):
        dim  = feat.shape[2]
        index  = index.unsqueeze(2).expand(index.shape[0], index.shape[1], dim)
        feat = feat.gather(1, index)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def transpose_and_gather_feat(self, feat, index):
        '''
        feat: B, H=32, W=32, C=5
        index: B, N
        '''
        feat = feat.view(feat.shape[0], -1, feat.shape[3]).contiguous()  # feat: B, H*W, C=5
        feat = self._gather_feat(feat, index)
        return feat

    def get_track_cls_loss_focal(self):
        cls_preds = self.forward_ret_dict['cls_preds']  # cls pred
        cls_labels = self.forward_ret_dict['cls_labels']  # cls label
        batch_size, height, width, channel = cls_preds.shape 
        cls_preds = cls_preds.permute(0, 3, 1 ,2).contiguous()
        cls_labels = cls_labels.unsqueeze(1)
        cls_preds_out = torch.clamp(cls_preds.sigmoid_(), min=1e-4, max=1-1e-4)
        cls_loss_src = self.cls_loss_func(cls_preds_out, cls_labels)  # [N, M]
        cls_loss = cls_loss_src.sum()

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.CLS_WEIGHTS
        tb_dict = {'track_loss_cls': cls_loss.item()}
        return cls_loss, tb_dict

    def get_track_reg_loss(self):
        loss_weights = self.model_cfg.LOSS_CONFIG.REG_WEIGHTS
        ind_label = self.forward_ret_dict['ind_labels'] # (B, N)
        reg_mask = self.forward_ret_dict['reg_mask'] # (B, N)
        label_value = self.forward_ret_dict['reg_labels'] # (B, N, 5)

        ry_pred = self.forward_ret_dict['ry_preds'] # [B, H, W, 5]
        z_pred = self.forward_ret_dict['z_preds'] # [B, H, W, 5]
        xy_pred = torch.zeros((ry_pred.shape[0], ry_pred.shape[1], ry_pred.shape[2], 2)).to(ry_pred.device)
        whl_pred = self.forward_ret_dict['whl_preds']  # [B, H, W, 5]
        reg_pred = torch.cat((xy_pred, z_pred, whl_pred, ry_pred),dim=-1)
        label_value[:, :, :2] = 0

        pred_value = self.transpose_and_gather_feat(reg_pred, ind_label)
        reg_mask = reg_mask.float().unsqueeze(2) 
        reg_loss = F.l1_loss(pred_value * reg_mask, label_value * reg_mask, reduction='none')
        loss = reg_loss / (reg_mask.sum() + 1e-4)
        reg_loss = (loss * torch.tensor(loss_weights,dtype=loss.dtype, device=loss.device)).sum()
        tb_dict = {'track_reg_loss': reg_loss.item()}

        return reg_loss, tb_dict

    def get_track_motion_loss(self):
        guss_radius = self.model_cfg.LOSS_CONFIG.GUSS_RADIUS
        loss_weights = self.model_cfg.LOSS_CONFIG.MT_WEIGHTS
        motion_map = self.forward_ret_dict['motion_map']
        motion_preds = self.forward_ret_dict['motion_preds']
        motion_mask = self.forward_ret_dict['motion_mask'].unsqueeze(-1)

        bs, x_shape, y_shape = motion_mask.shape[0], motion_mask.shape[1], motion_mask.shape[2]
        center_2d = torch.Tensor((x_shape / 2, y_shape / 2)).reshape(-1, 2)
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, x_shape - 1, x_shape),
                                        torch.linspace(0, y_shape - 1, y_shape))
        img_grid = torch.stack([grid_x, grid_y], -1).unsqueeze(0).repeat(bs, 1, 1, 1).reshape(-1, 2)
        depth2dc = torch.norm(img_grid.float() - center_2d.float(), dim=1, p=2, keepdim=False)
        gaussian_mask = torch.exp(-(depth2dc ** 2) / (2 * guss_radius)).reshape(bs, x_shape, y_shape, 1).to(motion_map.device)

        motion_preds = motion_preds * motion_mask
        motion_loss = self.mt_loss_func(motion_preds, motion_map)
        if self.model_cfg.LOSS_CONFIG.RIGID_OBJECT:
            motion_loss = motion_loss * gaussian_mask
        loss = motion_loss / (motion_mask.sum() + 1e-4)
        motion_loss = (loss * torch.tensor(loss_weights, dtype=loss.dtype, device=loss.device)).sum()
        tb_dict = {'track_motion_loss': motion_loss.item()}

        return motion_loss, tb_dict

    def get_track_corner_loss(self):
        corner_loss = self.forward_ret_dict['corner_loss'] * self.model_cfg.LOSS_CONFIG.COR_WEIGHTS #
        tb_dict = {'track_corner_loss': corner_loss.item()}

        return corner_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_track_cls_loss_focal()
        reg_loss, reg_tb_dict = self.get_track_reg_loss()
        motion_loss, motion_tb_dict = self.get_track_motion_loss()
        corner_loss, corner_tb_dict = self.get_track_corner_loss()
        tb_dict.update(reg_tb_dict)
        tb_dict.update(motion_tb_dict)
        tb_dict.update(corner_tb_dict)

        track_loss = cls_loss + reg_loss + motion_loss + corner_loss
        tb_dict['track_loss'] = track_loss.item()
        return track_loss, tb_dict

    def max_nms(self, heatmap, pool_size=3):
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(heatmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == heatmap).float()
        return heatmap * keep

    def topk_score(self,scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.view(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), index).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), index).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), index).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    @torch.no_grad()
    def generate_val_box(self, batch_dict):
        xy_preds = batch_dict['xy_preds'] # res_x, res_y, z, sinry, cosry
        z_preds = batch_dict['z_preds'] # res_x, res_y, z, sinry, cosry
        ry_preds = batch_dict['ry_preds'] # res_x, res_y, z, sinry, cosry
        cls_preds = batch_dict['cls_preds'].permute(0,3,1,2)
        whl = batch_dict['object_dim'] # B, 3
        batch_size = whl.shape[0]
        fmap = self.max_nms(cls_preds.sigmoid())

        scores, index, clses, ys, xs = self.topk_score(fmap, K=1)

        xy_preds = self.transpose_and_gather_feat(xy_preds, index)
        z_preds = self.transpose_and_gather_feat(z_preds, index)
        ry_preds = self.transpose_and_gather_feat(ry_preds, index)

        xy_preds = xy_preds.view(batch_size, -1)
        z_preds = z_preds.view(batch_size, -1)
        ry_preds = ry_preds.view(batch_size, -1)

        xs = xs.view(batch_size, 1) + xy_preds[:, 0:1]
        ys = ys.view(batch_size, 1) + xy_preds[:, 1:2]
        xs = xs * self.feature_map_stride * self.voxel_size[0] + self.pc_range[0]
        ys = ys * self.feature_map_stride * self.voxel_size[1] + self.pc_range[1]

        ry_pred = torch.atan2(ry_preds[:,0:1], ry_preds[:,1:2])

        final_box = torch.cat((xs, ys, z_preds, whl, ry_pred),dim=1)
        batch_dict['predict_box'] = final_box
        return batch_dict

        