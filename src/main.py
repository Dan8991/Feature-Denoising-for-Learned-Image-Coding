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

torch.backends.cudnn.benchmarks = True
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_dir = os.path.join("..", "dataset")

wandb.init(project="features-optimization", mode="online")
config = wandb.config

epochs = config.epochs
n_reps = config.n_reps
qp = config.qp
lr = config.lr
image_id = config.image_id
dataset = config.dataset
codec = config.codec
COMPUTE_REFERENCES = False

wandb.define_metric("psnr", summary="max")
allowed_codecs = ["Cheng2020", "JAHP", "MSH", "SH", "FP"]
if codec not in allowed_codecs:
    raise Exception(f"Unknown Codec {codec}, accepted values are: Cheng2020,"
                    " JAHP, MSH, SH, FP")

if image_id == -1:
    if dataset == "kodak":
        all_imgs = images_in_path(os.path.join(dataset_dir, "kodak"))
    else:
        all_imgs = glob(os.path.join(dataset_dir, "clic") + "/*.png")
else:
    if dataset == "kodak":
        all_imgs = [os.path.join(
            dataset_dir,
            "kodak",
            f"kodim{image_id:02}.png"
        )]
    else:
        all_imgs = glob(
            os.path.join(dataset_dir, "clic") + "/*.png"
        )[image_id:image_id + 1]

plot_path = os.path.join("..", "plot_info", "plot.json")
plot_dict = {}
if os.path.exists(plot_path):
    with open(plot_path, "r") as f:
        plot_dict = json.load(f)

plot_dict = evaluate_methods(
    all_imgs,
    plot_dict,
    COMPUTE_REFERENCES
)
with open(plot_path, "w") as f:
    json.dump(plot_dict, f, indent=2)

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

    dataset = ImageDataset(all_imgs, device)

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

        val_losses, sizes, psnrs = validation_step(
            feature_optimizer,
            dataset,
            ys,
            zs,
            shapes,
            ys_orig,
            xs_hat_orig
        )
        feature_optimizer = feature_optimizer.to(device)
        if sum(val_losses) < best_loss:
            best_loss = sum(val_losses)
            best_psnr = psnrs
            best_size = sizes
    parameters_dict[qpref] = parameters
    sizes_fo.append(best_size)
    psnrs_fo.append(best_psnr)

with open(os.path.join(wandb.run.dir, "parameters.pickle"), "wb") as f:
    pickle.dump(parameters_dict, f)
wandb.save(os.path.join(wandb.run.dir, "parameters.pickle"))
psnrs_fo = list(np.array(psnrs_fo).T)
sizes_fo = list(np.array(sizes_fo).T)
for image_path, psnr, rate in zip(all_imgs, psnrs_fo, sizes_fo):
    refine_str = codec + " + refinement"
    if  refine_str not in plot_dict[image_path]:
        plot_dict[image_path][refine_str] = {}
    plot_dict[image_path][refine_str]["psnr"] = [float(p) for p in psnr]
    plot_dict[image_path][refine_str]["rate"] = [float(s) for s in rate]

with open(plot_path, "w") as f:
    json.dump(plot_dict, f, indent=2)
with open(os.path.join(wandb.run.dir, "plot.json"), "w") as f:
    json.dump(plot_dict, f, indent=2)

wandb.save(os.path.join(wandb.run.dir, "plot.json"))

plot_rate_distortion(
    all_imgs,
    plot_dict
)
