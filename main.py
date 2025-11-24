import os
import cv2
import matplotlib.pyplot as plt
from src.pipeline import ChronoVisionPipeline


def main():
    # Setup paths
    COLORIZER_WEIGHTS = "models/main_efficientnet-b2_best.pt"
    GFPGAN_WEIGHTS = "models/GFPGANv1.3.pth"
    ESRGAN_WEIGHTS = "models/RealESRGAN_x4plus.pth"
    
    inputs_path = "data/test_inputs"
    outputs_path = "data/test_outputs"
    
    # Ensure output directory exists
    os.makedirs(outputs_path, exist_ok=True)

    # Initialize Pipeline
    pipeline = ChronoVisionPipeline(COLORIZER_WEIGHTS, GFPGAN_WEIGHTS, ESRGAN_WEIGHTS)

    # Get all image files from inputs directory
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(inputs_path)
                   if f.lower().endswith(image_extensions)]

    print(f"Found {len(image_files)} images to process")

    # Iterate over all images
    for image_name in image_files:
        print(f"Processing: {image_name}")

        try:
            # Run pipeline
            original, colorized, restored = pipeline.run(os.path.join(inputs_path, image_name))

            # Get filename without extension
            base_name = os.path.splitext(image_name)[0]

            # Display Results (Optional: Comment out if running in headless environment)
            plt.figure(figsize=(15, 8))

            plt.subplot(1, 3, 1)
            plt.title("Input (Grayscale)")
            plt.imshow(original)

            plt.subplot(1, 3, 2)
            plt.title("Stage 1: U-Net Colorization")
            plt.imshow(colorized)

            plt.subplot(1, 3, 3)
            plt.title("Stage 2: GFPGAN + RealESRGAN")
            plt.imshow(restored)

            plt.tight_layout()
            plt.show()

            # Save outputs
            cv2.imwrite(os.path.join(outputs_path, f"{base_name}_colorized.jpg"),
                       cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(outputs_path, f"{base_name}_restored.jpg"),
                       cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")

    print("All images have been processed")


if __name__ == "__main__":
    main()
