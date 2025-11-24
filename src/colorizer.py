import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp


class Colorizer:
    def __init__(
        self,
        model_path,
        encoder="efficientnet-b2",
        device=None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Initialize model structure
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,  # We load custom weights
            in_channels=3,  # Grayscale input (L channel)
            classes=2  # Output (ab channels)
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()


    def preprocess(self, img_rgb):
        # Convert to Lab
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        L = img_lab[:, :, 0]

        # Save the original dimensions
        original_h, original_w = L.shape

        # Resize the input image
        input_size = (224, 224)
        L_resized = cv2.resize(L, input_size)

        # Normalize
        L_norm = L_resized.astype(np.float32) / 255.0
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0) # (1, H, W)

        # Repeat the single L channel to imitate a 3-channel input
        L_tensor = L_tensor.repeat(3, 1, 1)  # (3, H, W)

        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        L_tensor = (L_tensor - mean) / std

        return L_tensor.unsqueeze(0).to(self.device), L, (original_h, original_w)


    def predict(self, img_rgb):
        """
        Input: RGB image (even if it's grayscale, read it as RGB)
        Output: Colorized RGB numpy image
        """
        tensor_input, L_original, (h, w) = self.preprocess(img_rgb)

        with torch.no_grad():
            ab_pred = self.model(tensor_input) * 1.05

        # Post-process
        ab_pred = ab_pred.cpu().squeeze(0).numpy().transpose(1, 2, 0)

        # Resize ab channels back to the original image size
        ab_pred = cv2.resize(ab_pred, (w, h))

        # Denormalize ab
        ab_pred = (ab_pred * 128.0 + 128.0).astype(np.uint8)

        # Combine with the original L channel
        lab_out = np.concatenate((L_original[:, :, np.newaxis], ab_pred), axis=2)
        rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB)

        return rgb_out
