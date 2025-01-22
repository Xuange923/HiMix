from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import ViTImageProcessor, ViTModel

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('vit-base')
class CLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = ViTModel(cfg)
        self._image_processor = ViTImageProcessor.from_pretrained(cfg.model_name_or_path)
  

