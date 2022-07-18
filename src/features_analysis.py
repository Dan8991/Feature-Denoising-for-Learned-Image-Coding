import pickle
import os
import numpy as np
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch
from compressai.zoo import cheng2020_anchor
from architecture import Cheng2020AnchorOptimized
import wandb

wandb_run = {
    "Cheng2020": "dan8991/nerf-images/3k4l54b1",
    "JAHP": "dan8991/nerf-images/3ppnzb5b",
    "MSH": "dan8991/nerf-images/ra91wc3g",
    "SH": "dan8991/nerf-images/6oy5ww69",
    "FP": "dan8991/nerf-images/mpbtbzu6"
}

wandb.init(project="features-optimization", mode="offline")
wandb.restore("parameters.pickle", run_path=wandb_run["Cheng2020"])

with open(os.path.join(wandb.run.dir, "parameters.pickle"), "rb") as f:
    parameters = pickle.load(f)

qpref = 6
lambdas = [0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045]
model = cheng2020_anchor(quality=qpref, pretrained=True)
feature_optimizer = Cheng2020AnchorOptimized(
    model.N,
    lambdas[qpref - 1]
).cpu()
feature_optimizer.load_state_dict(model.state_dict())
params = parameters[qpref]
with torch.no_grad():
    feature_optimizer.eval()
    model.eval()
    dataset_dir = os.path.join("..", "dataset")
    all_imgs = [os.path.join(
        dataset_dir,
        "kodak",
        f"kodim{19:02}.png"
    )]
    i = 10
    x = read_image(all_imgs[0]).unsqueeze(0).float()/255
    plt.imshow(x.squeeze(0).numpy().transpose(1, 2, 0))
    plt.xticks(color="w")
    plt.yticks(color="w")
    plt.tick_params(bottom=False)
    plt.tick_params(left=False)
    plt.tight_layout()
    plt.savefig("../plots/original.png")
    y_orig = model.g_a(x)
    z_orig = model.h_a(y_orig)
    res = model.compress(x)
    size_standard = sum(len(s[0]) for s in res["strings"])
    print(size_standard)
    x_hat = model.decompress(res["strings"], res["shape"])["x_hat"]
    mse = ((x-x_hat) ** 2).mean()
    print(10 * np.log10(1/mse))
    plt.imshow(x_hat.squeeze(0).numpy().transpose(1, 2, 0))
    plt.xticks(color="w")
    plt.yticks(color="w")
    plt.tick_params(bottom=False)
    plt.tick_params(left=False)
    plt.tight_layout()
    # plt.savefig("../plots/standard.png")
    # plt.show()
    y = params[0][0]
    z = params[1][0]
    h, w = x.shape[2:]
    y = y.cpu().reshape((y_orig.shape))
    z = z.cpu().reshape((z_orig.shape))
    res = feature_optimizer.compress(y, z)
    size_fd = sum(len(s[0]) for s in res["strings"])
    print(size_fd)
    print(1 - size_fd / size_standard)
    x_hat = feature_optimizer.decompress(res["strings"], res["shape"])["x_hat"]
    plt.imshow(x_hat.squeeze(0).numpy().transpose(1, 2, 0))
    plt.xticks(color="w")
    plt.yticks(color="w")
    plt.tick_params(bottom=False)
    plt.tick_params(left=False)
    plt.tight_layout()
    # plt.savefig("../plots/fd.png")
    # plt.show()
    mse = ((x - x_hat) ** 2).mean()
    print(10 * np.log10(1/mse))

    y = y.round().squeeze(0)
    y_orig = y_orig.round().squeeze(0)
    delta_y = (y != y_orig).sum(axis=0)
    z = z.round().squeeze(0)
    z_orig = z_orig.round().squeeze(0)
    delta_z = (z != z_orig).sum(axis=0)
    plt.imshow(delta_y, cmap="Blues")
    plt.xticks(color="w")
    plt.yticks(color="w")
    plt.tick_params(bottom=False)
    plt.tick_params(left=False)
    plt.colorbar()
    plt.savefig(f"../plots/heatmap_{qpref}.pdf", bbox_inches="tight")
    plt.show()
