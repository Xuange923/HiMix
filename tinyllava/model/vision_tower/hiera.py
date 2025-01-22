# source: https://hf-mirror.com/facebook/hiera-base-224-hf
from transformers import AutoImageProcessor, HieraModel


from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('hiera')
class HieraVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = HieraModel(cfg)
        self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path)
  

