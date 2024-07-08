from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('images/dog.jpg')
image_rgb = image.convert('RGB')
image_np = np.array(image_rgb)

# Lightness method


def lightness_method(image):
    max_rgb = np.max(image, axis=2)
    min_rgb = np.min(image, axis=2)
    gray_image = ((max_rgb + min_rgb) / 2).astype(np.uint8)
    print(gray_image[0, 0])
    return gray_image

# Average Method


def average_method(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    print(gray_image[0, 0])
    return gray_image

# Luminosity Method


def luminosity_method(image):
    gray_image = (0.21 * image[:, :, 0] + 0.72 * image[:,
                  :, 1] + 0.07 * image[:, :, 2]).astype(np.uint8)
    print(gray_image[0, 0])
    return gray_image


if __name__ == "__main__":

    gray_lightness = lightness_method(image_np)
    gray_average = average_method(image_np)
    gray_luminosity = luminosity_method(image_np)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(image_rgb)
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(gray_lightness, cmap='gray')
    axs[0, 1].set_title('Lightness Method')

    axs[1, 0].imshow(gray_average, cmap='gray')
    axs[1, 0].set_title('Average Method')

    axs[1, 1].imshow(gray_luminosity, cmap='gray')
    axs[1, 1].set_title('Luminosity Method')

    plt.show()
