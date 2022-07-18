import os
import json
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
from utils import images_in_path


def plot_rate_distortion(
    all_imgs,
    dictionary
):
    # plotting results
    plt.figure()

    image_shapes = [read_image(image_path).shape for image_path in all_imgs]
    pixel_sizes = [np.product(shape[-2:]) for shape in image_shapes]
    # getting the keys for one of the images
    for key in dictionary[all_imgs[0]]:
        sizes = np.mean([
            np.array(dictionary[image_path][key]["rate"]) /
            (1 if "+" in key else num_pixels / 8)
            for image_path, num_pixels in zip(all_imgs, pixel_sizes)
        ], axis=0)
        psnrs = np.mean(
            [dictionary[image_path][key]["psnr"] for image_path in all_imgs],
            axis=0
        )

        plt.plot(
            sizes,
            psnrs,
            label=key
        )

    plt.xlabel("bpp")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_path = os.path.join("..", "plot_info", "plot.json")
    with open(plot_path, "r") as f:
        plot_dict = json.load(f)
    dataset_dir = os.path.join("..", "dataset")
    images = images_in_path(os.path.join(dataset_dir, "kodak"))
    images = [os.path.join(dataset_dir, "kodak", f"kodim{1:02}.png")]
    plot_rate_distortion(
        images,
        plot_dict
    )
