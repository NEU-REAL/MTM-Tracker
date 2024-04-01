from .track3d_template import Track3DTemplate
from .mtm_track import MTM_Track
__all__ = {
    'Track3DTemplate': Track3DTemplate,
    'MTM_Track': MTM_Track,
}

def build_tracker(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
