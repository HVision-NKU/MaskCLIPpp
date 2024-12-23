from detectron2.utils.registry import Registry

TEXT_ENCODER_REGISTRY = Registry("TEXT_ENCODER")

def build_text_encoder(cfg):
    text_encoder_name = cfg.get("NAME", "none")
    if text_encoder_name.lower() == "none":
        return None
    text_encoder = TEXT_ENCODER_REGISTRY.get(text_encoder_name)(cfg)
    return text_encoder