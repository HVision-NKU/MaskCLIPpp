from detectron2.utils.registry import Registry

PSM_REGISTRY = Registry("PSM")

def build_psm(cfg, input_shape, template_dim):
    psm_name = cfg.NAME
    psm = PSM_REGISTRY.get(psm_name)(cfg, input_shape, template_dim)
    return psm