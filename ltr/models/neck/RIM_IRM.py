import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from ltr.ops.deformattn.modules import MSDeformAttn
import copy
import math
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list
        not_mask = torch.ones((x.shape[0], x.shape[2], x.shape[3])).bool()
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None].to(x.device) / dim_t
        pos_y = y_embed[:, :, :, None].to(x.device) / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class DeformAttnModule(nn.Module):
    def __init__(self, deformattn_layers, num_layers):
        super().__init__()
        self.deformattn_layers = self._get_clones(deformattn_layers, num_layers)
        self.num_layers = num_layers

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.deformattn_layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class DeformAttnLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    # input: N,C,Hin,Win
    # grid: N,Hout,Wout,2
    # output: N,C,Hout,Wout
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=1, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)  # batch, ht, wd, 1, ht, wd

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)  # batch*h1*w1, 1, ht, wd

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)  # B, D, W*H
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        self.corr_levels = 1
        self.corr_radius = 4
        cor_planes = self.corr_levels * (2*self.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=256, input_dim=128 + 256)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        delta_flow = self.flow_head(motion_features)

        return net, delta_flow

class FeatureInteraction_MotionRefinement(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.feats_list = model_cfg.NECK_OUTCHANNEL
        self.num_neck_features = model_cfg.NECK_OUTCHANNEL
        self.iters = 10

        self.update_block = BasicUpdateBlock()
        self.cls_layers = self.make_final_layers(model_cfg.NECK_OUTCHANNEL, model_cfg.CLS_FC)
        self.ad_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.encoder = nn.Sequential(nn.Conv2d(model_cfg.DeformAttn.D_MODEL*3, model_cfg.NECK_OUTCHANNEL, 3, padding=1),
                                     nn.BatchNorm2d(model_cfg.NECK_OUTCHANNEL),
                                     nn.ReLU(),
                                     nn.Conv2d(model_cfg.NECK_OUTCHANNEL, model_cfg.NECK_OUTCHANNEL, 3, padding=1),
                                     nn.BatchNorm2d(model_cfg.NECK_OUTCHANNEL),
                                     nn.ReLU())
        self.conv2d = torch.nn.Conv2d(model_cfg.DeformAttn.D_MODEL, model_cfg.DeformAttn.D_MODEL, 1)
        deformattn_layer = DeformAttnLayer(d_model = model_cfg.DeformAttn.D_MODEL,
                                           d_ffn = model_cfg.DeformAttn.DIM_FEEDFORWARD,
                                           dropout = model_cfg.DeformAttn.DROPOUT,
                                           n_levels=model_cfg.DeformAttn.N_LEVELS,
                                           n_heads = model_cfg.DeformAttn.N_HEADS,
                                           n_points = model_cfg.DeformAttn.N_POINTS)
        self.deformattn_module = DeformAttnModule(deformattn_layer, model_cfg.DeformAttn.NUM_ENCODER_LAYERS)
        self.level_embed = nn.Parameter(torch.Tensor(model_cfg.DeformAttn.N_LEVELS, model_cfg.DeformAttn.D_MODEL))
        self.pos_embed = PositionEmbeddingSine(model_cfg.DeformAttn.D_MODEL // 2)
        self.model_cfg = model_cfg
        self.corr_radius = 4

        self.hidden_dim = 128
        self.context_dim = 128

        self.voxel_size = model_cfg.VOXEL_SIZE
        self.up_ratio = model_cfg.UP_RATIO
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.cls_layers[-1].bias, -np.log((1 - pi) / pi))
        normal_(self.level_embed)

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

    def initialize_flow(self, feat):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = feat.shape
        coords0 = coords_grid(N, H, W, device=feat.device)
        coords1 = coords_grid(N, H, W, device=feat.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def max_nms(self, heatmap, pool_size=3):
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(heatmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == heatmap).float()
        return heatmap * keep

    def _gather_feat(self, feat, index, mask=None):
        dim  = feat.shape[2]
        index  = index.unsqueeze(2).expand(index.shape[0], index.shape[1], dim)
        feat = feat.gather(1, index)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

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

    def upflow8(self, flow, mode='bilinear', up_ratio=2):
        new_size = (up_ratio * flow.shape[2], up_ratio * flow.shape[3])
        return up_ratio * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def fusion(self, feats):
        weighted_feats = []
        for feat in feats:
            avg_feat = self.ad_avg_pool(feat)
            weight = torch.sigmoid(self.conv2d(avg_feat))
            feat = feat * weight + feat
            weighted_feats.append(feat)
        weighted_feats = torch.cat(weighted_feats, dim=1)
        weighted_feats = self.encoder(weighted_feats)
        return weighted_feats

    def forward(self, batch_dict):
        # ============= Reciprocating Interaction Module ============= #
        sp_feats = batch_dict['search_spatial_features_2d']  # search point feature
        tp_feats = batch_dict['template_spatial_features_2d']  # template point feature
        BS, C, X, Y = sp_feats[0].shape

        src_flatten = []
        spatial_shapes = []
        for lvl, (sp_feat, tp_feat) in enumerate(zip(sp_feats, tp_feats)):
            bs, c, x, y = sp_feat.shape
            spatial_shape = (x, y)
            src = torch.cat((sp_feat, tp_feat), dim=1).flatten(2).transpose(1, 2)
            src_flatten.append(src)
            spatial_shapes.append(spatial_shape)

        src_flatten = torch.cat(src_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones((BS, 3, 2),dtype=src_flatten.dtype, device=src_flatten.device)

        deform_feats = self.deformattn_module(src_flatten, spatial_shapes, level_start_index, valid_ratios)

        deform_feats = deform_feats[:, level_start_index[0]: level_start_index[1], :]
        deform_feats = deform_feats.transpose(1, 2).reshape(BS, C*2, X, Y)
        deform_sp_feats = deform_feats[:, :C, :, :]
        deform_tp_feats = deform_feats[:, C:, :, :]

        cls_preds = self.cls_layers(deform_sp_feats)
        batch_dict['cls_preds'] = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        batch_dict['fusion_feature'] = deform_sp_feats

        # ============= Iterative Refinement Module ============= #
        # ---------- init flow ----------
        corner_off = batch_dict['predicted_box_corner_offset'][:, 0, :].reshape(BS, 3, 9)
        center_off = torch.mean(corner_off, dim=2)[:, :2] / self.voxel_size / 8
        center_off_x = center_off[:, 0:1]
        center_off_y = center_off[:, 1:]

        init_flow_x = center_off_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, X, Y)
        init_flow_y = center_off_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, X, Y)
        init_flow = torch.ones((BS, 2, X, Y), device=deform_sp_feats.device)
        init_flow[:, :0, :, :] = init_flow[:, :0, :, :] * init_flow_x
        init_flow[:, 1:, :, :] = init_flow[:, 1:, :, :] * init_flow_y

        # ---------- optical flow ----------
        hdim = self.hidden_dim
        cdim = self.context_dim

        corr_fn = CorrBlock(deform_tp_feats, deform_sp_feats, num_levels=1, radius=self.corr_radius)
        coords0, coords1 = self.initialize_flow(deform_tp_feats)
        coords1 = coords1 + init_flow

        net, inp = torch.split(deform_tp_feats, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        batch_dict['motion_preds_aux'] = {}
        flow_up = None
        for itr in range(self.iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)
            flow = coords1 - coords0

            net, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            flow_up = self.upflow8(coords1 - coords0, up_ratio=self.up_ratio)

            if itr < self.iters - 1:
                batch_dict['motion_preds_aux'].update({'motion_preds_' + str(itr): flow_up.permute(0, 2, 3, 1)})

        batch_dict['motion_preds'] = flow_up.permute(0, 2, 3, 1)

        return batch_dict
