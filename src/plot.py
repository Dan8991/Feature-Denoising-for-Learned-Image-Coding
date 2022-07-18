import matplotlib
from torchvision.io import read_image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from architecture import schedule
from utils import images_in_path
from plot_utils import plot_rate_distortion

plot_lr = False
plot_results = True

def bj_delta(R1, PSNR1, R2, PSNR2, mode=0, poly_exp=3):
    '''
    Computes the Bjontegaard delta rate for the given plots
    Parameters:
        R1 (np.ndarray): rates for the new codec
        PSNR1 (np.ndarray): PSNR for the new codec
        R1 (np.ndarray): rate for the reference codec
        PSNR1 (np.ndarray): PSNR for the reference codec
        mode (int): 0 returns delta rate 1 returns delta PSNR
        poly_exp (int): defines the degree of the fitted polynomial 
    Returns:
        avg_diff (float): delta value
        
    '''
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # find integral
    if mode == 0:
        # least squares polynomial fit
        p1 = np.polyfit(lR1, PSNR1, poly_exp)
        p2 = np.polyfit(lR2, PSNR2, poly_exp)

        # integration interval
        min_int = max(min(lR1), min(lR2))
        max_int = min(max(lR1), max(lR2))

        # indefinite integral of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_diff = (int2-int1)/(max_int-min_int)
    else:
        # rate method: sames as previous one but with inverse order
        p1 = np.polyfit(PSNR1, lR1, poly_exp)
        p2 = np.polyfit(PSNR2, lR2, poly_exp)

        # integration interval
        min_int = max(min(PSNR1), min(PSNR2))
        max_int = min(max(PSNR1), max(PSNR2))

        # indefinite interval of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff

if plot_lr:
    x = range(10000)
    y = []
    matplotlib.rc("font", **{"size": 13})
    for sample in x:
        y.append(1e-6 * schedule(1.01)(sample))
    plt.plot(x, y)
    plt.xlabel("Optimization Step")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

colors = ["purple", "red", "green", "blue", "orange"]

if plot_results:
    plt.figure(figsize=(6, 4))
    codecs = ["Cheng2020", "JAHP", "MSH", "SH", "FP"]
    plot_path = os.path.join("..", "plot_info", "plot_fixed.json")
    with open(plot_path, "r") as f:
        plot_dict = json.load(f)
    dataset_dir = os.path.join("..", "dataset")
    images = images_in_path(os.path.join(dataset_dir, "kodak"))
    new_dict = {}
    image_shapes = [read_image(image_path).shape for image_path in images]
    pixel_sizes = [np.product(shape[-2:]) for shape in image_shapes]
    for key in plot_dict[images[0]]:
        matplotlib.rc("font", **{"size": 13})
        sizes = np.mean([
            np.array(plot_dict[image_path][key]["rate"]) /
            (1 if "+" in key else num_pixels / 8)
            for image_path, num_pixels in zip(images, pixel_sizes)
        ], axis=0)
        psnrs = np.mean(
            [plot_dict[image_path][key]["psnr"] for image_path in images],
            axis=0
        )
        new_dict[key] = (sizes, psnrs)
    for i, codec in enumerate(codecs):
        keys = [key for key in plot_dict[images[0]].keys() if key[:len(codec)] == codec]
        size_orig = new_dict[keys[0]][0]
        psnr_orig = new_dict[keys[0]][1]
        size_imp = new_dict[keys[1]][0]
        psnr_imp = new_dict[keys[1]][1]
        plt.plot(size_orig, np.interp(size_orig, size_imp, psnr_imp) - psnr_orig, c=colors[i], label=keys[0], linestyle="--")
        delta_rate = bj_delta(*new_dict[keys[0]], *new_dict[keys[1]], mode=1)
        delta_psnr = bj_delta(*new_dict[keys[0]], *new_dict[keys[1]], mode=0)
        print(f"model: {keys[0]}, rate: {delta_rate}, psnr {delta_psnr}")

    delta_rate = bj_delta(*new_dict["VVC"], *new_dict["Cheng2020 + refinement"], mode=1)
    delta_psnr = bj_delta(*new_dict["VVC"], *new_dict["Cheng2020 + refinement"], mode=0)
    print(f"model: VVC vs Cheng + ref, rate: {delta_rate}, psnr {delta_psnr}")
    delta_rate = bj_delta(*new_dict["VVC"], *new_dict["Cheng2020"], mode=1)
    delta_psnr = bj_delta(*new_dict["VVC"], *new_dict["Cheng2020"], mode=0)
    print(f"model: VVC vs Cheng, rate: {delta_rate}, psnr {delta_psnr}")

    plt.legend()
    plt.xlabel("Rate(bpp)")
    plt.ylabel("Delta PSNR (refined - original)")
    plt.tight_layout()
    plt.grid()
    plt.savefig("../plots/delta_plots.pdf")
    plt.show()


    plt.figure(figsize=(6, 4.5))
    plt.plot(
        new_dict["VVC"][0],
        new_dict["VVC"][1],
        c="black",
        linestyle="-",
        label="VVC"
    )

    for i, codec in enumerate(codecs):
        plot_type = 0 #0 = standard approach, 1 = FD
        keys = [key for key in plot_dict[images[0]].keys() if key[:len(codec)] == codec]
        plt.plot(
            new_dict[keys[plot_type]][0],
            new_dict[keys[plot_type]][1],
            c=colors[i],
            linestyle="--",
            label=keys[plot_type].replace("refinement", "FD")
        )

    plt.xlim([0, 2.4])
    plt.legend()
    plt.grid()
    plt.xlabel("Rate(bpp)")
    plt.ylabel("PSNR(dB)")
    plt.tight_layout()
    plt.savefig("../plots/VVC_standard.pdf")
    plt.show()

    delta_rates = [[] for _ in codecs]
    delta_psnrs = [[] for _ in codecs]
    for image in images:
        num_pixels = np.product(read_image(image).shape[-2:])
        for i, codec in enumerate(codecs):
            keys = [key for key in plot_dict[image].keys() if key[:len(codec)] == codec]
            keys.sort(key=lambda x: len(x))
            size_orig = np.array(plot_dict[image][keys[0]]["rate"]) * 8 / num_pixels
            psnr_orig = plot_dict[image][keys[0]]["psnr"]
            size_imp = plot_dict[image][keys[1]]["rate"]
            psnr_imp = plot_dict[image][keys[1]]["psnr"]
            delta_rate = bj_delta(size_orig, psnr_orig, size_imp, psnr_imp, mode=1)
            delta_psnr = bj_delta(size_orig, psnr_orig, size_imp, psnr_imp, mode=0)
            delta_rates[i].append(delta_rate)
            delta_psnrs[i].append(delta_psnr)

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    mean_delta_rates = np.mean(delta_rates, axis=1)
    mean_delta_psnrs = np.mean(delta_psnrs, axis=1)
    min_delta_rates = np.min(delta_rates, axis=1).reshape(1, -1)
    min_delta_psnrs = np.min(delta_psnrs, axis=1).reshape(1, -1)
    max_delta_rates = np.max(delta_rates, axis=1).reshape(1, -1)
    max_delta_psnrs = np.max(delta_psnrs, axis=1).reshape(1, -1)
    error_rate = np.concatenate([
        mean_delta_rates.reshape(1, -1) - min_delta_rates,
        max_delta_rates - mean_delta_rates.reshape(1, -1)
    ], axis=0)
    error_psnr = np.concatenate([
        mean_delta_psnrs.reshape(1, -1) - min_delta_psnrs,
        max_delta_psnrs - mean_delta_psnrs.reshape(1, -1)
    ], axis=0)
    axs[0].bar(codecs, mean_delta_rates, color=colors, alpha=0.7)
    axs[1].bar(codecs, mean_delta_psnrs, color=colors, alpha=0.7)
    axs[0].errorbar(
        codecs,
        np.mean(delta_rates, axis=1),
        xerr = None,
        yerr = error_rate,
        c="black",
        fmt="none",
        capsize=10
    )
    axs[1].errorbar(
        codecs,
        np.mean(delta_psnrs, axis=1),
        yerr = error_psnr,
        c="black",
        fmt="none",
        capsize=10
    )
    axs[0].set_xticklabels(codecs, rotation=45)
    axs[1].set_xticklabels(codecs, rotation=45)
    axs[0].set_ylabel("BJ Rate (%)")
    axs[1].set_ylabel("BJ PSNR (dB)")
    plt.tight_layout()
    plt.savefig("../plots/bj_plots.pdf")
    plt.show()

    for i, codec in enumerate(codecs):
        rate, psnr = new_dict[codec]
        rate_FD, psnr_FD = new_dict[codec + " + refinement"]
        perc_rate = (rate_FD - rate) / rate * 100
        perc_psnr = (psnr_FD - psnr) / psnr * 100
        plt.figure(1, figsize=(6, 4.5))
        plt.plot(perc_rate, c=colors[i], linestyle="--", label=codec)
        plt.xlabel("QP")
        plt.ylabel("Rate difference (%)")
        plt.figure(2, figsize=(6, 4.5))
        plt.plot(perc_psnr, c=colors[i], linestyle="--", label=codec)
        plt.xlabel("QP")
        plt.ylabel("PSNR difference (%)")
    plt.figure(1, figsize=(6, 4.5))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../plots/rate_perc.pdf")
    plt.figure(2, figsize=(6, 4.5))
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("../plots/psnr_perc.pdf")
    plt.show()
