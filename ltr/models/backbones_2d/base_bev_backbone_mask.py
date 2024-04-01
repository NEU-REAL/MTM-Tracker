import numpy as np
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

class BaseBEVBackboneMask(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        self.down_stride = upsample_strides[0] * layer_strides[0]
        num_levels = len(layer_nums)

        tp_c_in_list = [input_channels + 1, *num_filters[:-1]]
        sr_c_in_list = [input_channels, *num_filters[:-1]]
        self.tp_blocks = nn.ModuleList()
        self.tp_deblocks = nn.ModuleList()
        self.sr_blocks = nn.ModuleList()
        self.sr_deblocks = nn.ModuleList()

        c_in = self.create_layer(self.tp_blocks, self.tp_deblocks, num_levels, tp_c_in_list, layer_strides, num_filters, layer_nums, upsample_strides,
                                 num_upsample_filters)
        _ = self.create_layer(self.sr_blocks, self.sr_deblocks, num_levels, sr_c_in_list, layer_strides, num_filters, layer_nums, upsample_strides,
                                 num_upsample_filters)

        self.project_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.CBR = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                                 nn.ReLU())

        self.num_bev_features = c_in
        self.feats_list = [c_in]

    def create_layer(self, blocks, deblocks, num_levels, c_in_list, layer_strides, num_filters, layer_nums, upsample_strides, num_upsample_filters):
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        return c_in

    def forward(self, batch_dict):
        search_spatial_features = batch_dict['search_spatial_features']
        template_spatial_features = batch_dict['template_spatial_features']
        template_bev_mask = batch_dict['template_bev_mask'].unsqueeze(1)
        template_spatial_features = torch.cat((template_spatial_features, template_bev_mask), dim=1)

        batch_dict['search_spatial_features_2d'] , batch_dict['template_spatial_features_2d'] \
            = self.forward_feature(search_spatial_features, template_spatial_features, self.sr_blocks, self.sr_deblocks, self.tp_blocks, self.tp_deblocks)
        batch_dict['spatial_features_stride'] *= self.down_stride

        return batch_dict

    def feat_attention(self, feat):
        feat_max, _ = torch.max(input=feat, dim=1, keepdim=True)
        feat_avg = torch.mean(input=feat, dim=1, keepdim=True)
        feat_concat = torch.cat((feat_max, feat_avg), dim=1)
        attention = torch.sigmoid(self.project_conv(feat_concat))
        return attention

    def process_multifeat(self, feats):
        new_feats = []
        for feat in feats:
            diffY = feats[0].size()[2] - feat.size()[2]
            diffX = feats[0].size()[3] - feat.size()[3]
            feat = F.pad(feat, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
            new_feats.append(feat)

        return new_feats

    def forward_feature(self, sr_spatial_features, tp_spatial_features, sr_blocks, sr_deblocks, tp_blocks, tp_deblocks):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        sr_ups = []
        tp_ups = []
        ret_dict = {}
        sr_x = sr_spatial_features
        tp_x = tp_spatial_features

        for i in range(len(sr_blocks)):
            sr_x = sr_blocks[i](sr_x)
            tp_x = tp_blocks[i](tp_x)

            stride = int(sr_spatial_features.shape[2] / sr_x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = sr_x

            sr_attention = self.feat_attention(sr_x)
            tp_attention = self.feat_attention(tp_x)

            sr_x = sr_x * tp_attention + sr_x
            tp_x = tp_x * sr_attention + tp_x

            sr_ups.append(sr_x)
            tp_ups.append(tp_x)

        return sr_ups, tp_ups