import torch.nn as nn

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:
        """

        spconv_x = batch_dict['search_sparse_features']
        spconv_t = batch_dict['template_sparse_features']
        batch_dict['search_spatial_features'] = self.forward_feature(spconv_x)
        batch_dict['template_spatial_features'] = self.forward_feature(spconv_t)
        batch_dict['spatial_features_stride'] = batch_dict['spconv_downstride']
        return batch_dict

    def forward_feature(self, sparse_feature):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        spatial_features = sparse_feature.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features