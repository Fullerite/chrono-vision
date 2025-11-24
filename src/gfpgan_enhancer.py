import torch
from gfpgan import GFPGANer

from src.realesrgan_enhancer import RealESRGANEnhancer


class GFPGANEnhancer:
    def __init__(
        self,
        model_path: str,
        bg_model_path: str,
        upscale: int = 4,
        tile: int = 256,
        device: str | None = None,
        bg_scale: int | None = None,
    ):
        """Wrapper around GFPGAN face restoration model with RealESRGAN background upsampler."""
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.upscale = upscale
        self.tile = tile
        self.bg_scale = bg_scale if bg_scale is not None else upscale

        # Background upsampler using the existing RealESRGANEnhancer
        self.bg_upsampler = RealESRGANEnhancer(
            model_path=bg_model_path,
            scale=self.bg_scale,
            tile=tile,
            device=self.device.type,
        ).enhancer

        # GFPGAN restorer
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.bg_upsampler,
            device=self.device,
        )

    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        """Run GFPGAN restoration on an image (np.ndarray). Returns restored image only."""
        if img is None:
            raise ValueError("Input image is None")

        _, _, restored_img = self.restorer.enhance(
            img,
            has_aligned=has_aligned,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        return restored_img
