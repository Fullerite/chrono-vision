import cv2
from src.colorizer import Colorizer
from src.gfpgan_enhancer import GFPGANEnhancer


class ChronoVisionPipeline:
    def __init__(
        self,
        colorizer_weights,
        gfpgan_weights,
        esrgan_weights,
        device=None
    ):
        print("Initializing ChronoVision Pipeline...")

        # Stage 1: Colorization
        self.colorizer = Colorizer(
            model_path=colorizer_weights,
            device=device
        )

        # Stage 2 & 3: Restoration & Upscaling (GFPGAN wraps RealESRGAN)
        self.restorer = GFPGANEnhancer(
            model_path=gfpgan_weights,
            bg_model_path=esrgan_weights,
            upscale=4,
            device=device
        )


    def run(self, image_path):
        # 1. Load Image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Colorize
        print("Running Stage 1: Colorization...")
        colorized = self.colorizer.predict(img)

        # 3. Restore
        print("Running Stage 2: Face Restoration & Upscaling...")
        final_output = self.restorer.enhance(
            colorized,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        return img, colorized, final_output
