from glob import glob
import os
import json
import random
import pickle
import numpy as np
from compressai.zoo import cheng2020_anchor, mbt2018, mbt2018_mean
from compressai.zoo import bmshj2018_hyperprior, bmshj2018_factorized
from torch.optim import Adam
import torch
import wandb
from architecture import Cheng2020AnchorOptimized, JAHPOptimized
from architecture import FactorizedPriorOptimized, schedule
from architecture import MeanScaleHyperpriorOptimized, ScaleHyperpriorOptimized
from architecture import produce_features, train_step, validation_step
from dataset import ImageDataset
from plot_utils import plot_rate_distortion
from utils import images_in_path, evaluate_methods
import matplotlib.pyplot as plt
from torchvision.utils import save_image

torch.backends.cudnn.benchmarks = True
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(project="features-optimization", mode="disabled")
config = wandb.config

epochs = config.epochs
n_reps = config.n_reps
qp = config.qp
lr = config.lr
codec = config.codec
input_path = config.input_path
compressed_path = config.compressed_path
output_path = config.output_path

wandb.define_metric("psnr", summary="max")
allowed_codecs = ["Cheng2020", "JAHP", "MSH", "SH", "FP"]
if codec not in allowed_codecs:
    raise Exception(f"Unknown Codec {codec}, accepted values are: Cheng2020,"
                    " JAHP, MSH, SH, FP")

psnrs_fo = []
sizes_fo = []

if qp < 0:
    if codec == "Cheng2020":
        qp_range = range(1, 7)
    else:
        qp_range = range(1, 9)
else:
    qp_range = range(qp, qp + 1)

psnrs_fo = []
sizes_fo = []
psnrs_orig = []
sizes_orig = []
best_bitstream = None
best_reconstruction = None
parameters_dict = {}
for qpref in qp_range:

    lambdas = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
    if codec == "Cheng2020":
        lambdas = [0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045]

        model = cheng2020_anchor(quality=qpref, pretrained=True)
        feature_optimizer = Cheng2020AnchorOptimized(
            model.N,
            lambdas[qpref - 1]
        )
    elif codec == "JAHP":
        model = mbt2018(quality=qpref, pretrained=True)
        feature_optimizer = JAHPOptimized(
            model.N,
            model.M,
            lambdas[qpref - 1]
        )
    elif codec == "MSH":
        model = mbt2018_mean(quality=qpref, pretrained=True)
        feature_optimizer = MeanScaleHyperpriorOptimized(
            model.N,
            model.M,
            lambdas[qpref - 1]
        )
    elif codec == "SH":
        model = bmshj2018_hyperprior(quality=qpref, pretrained=True)
        feature_optimizer = ScaleHyperpriorOptimized(
            model.N,
            model.M,
            lambdas[qpref - 1]
        )
    elif codec == "FP":
        model = bmshj2018_factorized(quality=qpref, pretrained=True)
        feature_optimizer = FactorizedPriorOptimized(
            model.N,
            model.M,
            lambdas[qpref - 1]
        )

    feature_optimizer.load_state_dict(model.state_dict())

    dataset = ImageDataset([input_path], device)

    ys, zs, ys_orig, xs_hat_orig, shapes = produce_features(
        dataset,
        feature_optimizer,
        codec=codec
    )

    ys = torch.nn.Parameter(ys.to(device))
    if codec != "FP":
        zs = torch.nn.Parameter(zs.to(device))
        zs.requires_grad = True
        parameters = [ys, zs]
    else:
        parameters = [ys]

    ys.requires_grad = True
    parameters_group = [
        {"params": parameters, "lr": lr},
        {"params": feature_optimizer.parameters(), "lr": 3e-3},
    ]
    opt = Adam(parameters_group)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        [schedule(1.01), lambda x: 1]
    )
    best_size = np.inf
    best_psnr = -np.inf
    best_loss = np.inf
    for epoch in range(epochs):
        train_step(
            feature_optimizer,
            ys,
            zs,
            shapes,
            opt,
            dataset,
            device=device,
            n_reps=n_reps,
            epoch=epoch,
            scheduler=scheduler,
            qp=qpref
        )

        val_losses, sizes, psnrs, size_o, psnr_o, rec_image, bitstream = validation_step(
            feature_optimizer,
            dataset,
            ys,
            zs,
            shapes,
            ys_orig,
            xs_hat_orig,
            get_orig=True
        )
        sizes_orig.append(size_o[0])
        psnrs_orig.append(psnr_o[0])
        feature_optimizer = feature_optimizer.to(device)
        if sum(val_losses) < best_loss:
            best_parameters = [param.clone() for param in parameters]
            best_loss = sum(val_losses)
            best_psnr = psnrs
            best_size = sizes
            best_bitstream = bitstream
            best_reconstruction = rec_image
    parameters_dict[qpref] = best_parameters
    sizes_fo.append(best_size)
    psnrs_fo.append(best_psnr)

    if compressed_path is not None:

        if qp < 0:
            final_compressed_path = compressed_path[:-4] + f"_qp_{qpref}.bin"
        else:
            final_compressed_path = compressed_path

        with open(final_compressed_path, "wb") as f:
            pickle.dump(best_bitstream, f)

    if output_path is not None:

        if qp < 0:
            final_output_path = output_path[:-4] + f"_qp_{qpref}.png"
        else:
            final_output_path = output_path

        save_image(best_reconstruction.float() / 255, final_output_path)

psnrs_fo = np.array(psnrs_fo)
sizes_fo = np.array(sizes_fo)
if len(psnrs_fo) > 1:
    plt.plot(sizes_fo, psnrs_fo, label="feature optimization")
    plt.plot(sizes_orig, psnrs_orig, label="original")
    plt.legend()
    plt.title("RD curve")
    plt.xlabel("Rate(bpp)")
    plt.ylabel("PSNR(dB)")
    plt.show()
