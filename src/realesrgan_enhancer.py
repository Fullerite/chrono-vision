import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class RealESRGANEnhancer:
    def __init__(
        self,
        model_path: str = "../models/RealESRGAN_x4plus.pth",
        scale: int = 4,
        tile: int = 256,
        device: str | None = None,
    ):
        """Wrapper around Real-ESRGAN super-resolution model."""
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scale = scale
        self.tile = tile
        self.model_path = model_path

        # Build model architecture
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

        # Initialize enhancer
        self.enhancer = RealESRGANer(
            scale=scale,
            model=self.model,
            model_path=model_path,
            tile=tile,
            device=self.device,
            half=(self.device.type == "cuda"),
        )

    def enhance(self, img):
        """Run Real-ESRGAN super-resolution on an image (np.ndarray)."""
        if img is None:
            raise ValueError("Input image is None")
        output, _ = self.enhancer.enhance(img, outscale=self.scale)
        return output
