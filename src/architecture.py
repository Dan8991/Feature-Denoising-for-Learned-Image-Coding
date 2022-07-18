import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from compressai.models import Cheng2020Anchor, MeanScaleHyperprior
from compressai.models import ScaleHyperprior, FactorizedPrior
from compressai.models import JointAutoregressiveHierarchicalPriors
import wandb


class Cheng2020AnchorOptimized(Cheng2020Anchor):

    def __init__(self, N, lambdaRD=0.1, **kwargs):
        super().__init__(N, **kwargs)
        self.lambdaRD = lambdaRD

        # making the model non trainable
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, y, z):

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(
            y,
            "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(
            y,
            scales_hat,
            means=means_hat
        )
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, y, z):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


class JAHPOptimized(JointAutoregressiveHierarchicalPriors):

    def __init__(self, N, M, lambdaRD=0.1, **kwargs):
        super().__init__(N, M, **kwargs)

        self.lambdaRD = lambdaRD

        # making the model non trainable
        for parameter in self.parameters():
            parameter.requires_grad = False
        # making the entropy estimator trainable since it just works as a
        # regularizer
        # for parameter in self.entropy_bottleneck.parameters():
            # parameter.requires_grad = True

    def forward(self, y, z):

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(
            y,
            "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(
            y,
            scales_hat,
            means=means_hat
        )
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, y, z):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


class MeanScaleHyperpriorOptimized(MeanScaleHyperprior):

    def __init__(self, N, M, lambdaRD=0.1, **kwargs):
        super().__init__(N, M, **kwargs)

        self.lambdaRD = lambdaRD

        # making the model non trainable
        for parameter in self.parameters():
            parameter.requires_grad = False
        # making the entropy estimator trainable since it just works as a
        # regularizer
        # for parameter in self.entropy_bottleneck.parameters():
            # parameter.requires_grad = True

    def forward(self, y, z):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(
            y,
            scales_hat,
            means=means_hat
        )
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, y, z):
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(
            y,
            indexes,
            means=means_hat
        )
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


class ScaleHyperpriorOptimized(ScaleHyperprior):

    def __init__(self, N, M, lambdaRD=0.1, **kwargs):
        super().__init__(N, M, **kwargs)

        self.lambdaRD = lambdaRD

        # making the model non trainable
        for parameter in self.parameters():
            parameter.requires_grad = False
        # making the entropy estimator trainable since it just works as a
        # regularizer
        # for parameter in self.entropy_bottleneck.parameters():
            # parameter.requires_grad = True

    def forward(self, y, z):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, y, z):
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


class FactorizedPriorOptimized(FactorizedPrior):

    def __init__(self, N, M, lambdaRD=0.1, **kwargs):
        super().__init__(N, M, **kwargs)

        self.lambdaRD = lambdaRD

        # making the model non trainable
        for parameter in self.parameters():
            parameter.requires_grad = False
        # making the entropy estimator trainable since it just works as a
        # regularizer
        # for parameter in self.entropy_bottleneck.parameters():
            # parameter.requires_grad = True

    def forward(self, y):
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def compress(self, y):
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}


def produce_features(
    dataset,
    model,
    codec="Cheng2020"
):

    model.eval()

    ys = []
    zs = []
    ys_orig = []
    xs_hat_orig = []
    shapes = []

    is_fp = codec == "FP"
    model = model.cpu()
    for image in dataset.X:
        image = image.unsqueeze(0).cpu()
        with torch.no_grad():
            y = model.g_a(image)
            if codec != "SH" and not is_fp:
                z = model.h_a(y)
            elif codec == "SH":
                z = model.h_a(torch.abs(y))
            ys.append(y.reshape(y.shape[:2] + (-1,)))
            if not is_fp:
                zs.append(z.reshape(z.shape[:2] + (-1,)))
            shapes.append((y.shape, ) if is_fp else (y.shape, z.shape))
            if is_fp:
                latent = model.compress(y)
            else:
                latent = model.compress(y, z)
            ys_orig.append(latent)
            xs_hat_orig.append(
                model.decompress(
                    latent["strings"],
                    latent["shape"]
                )["x_hat"]
            )
    lengths = [y.shape[-1] for y in ys]
    ys = [
        torch.cat([
            y, torch.zeros(y.shape[:-1] + (max(lengths) - y.shape[-1],))
        ], axis=-1) for y in ys
    ]

    ys = torch.cat(ys, axis=0)

    if is_fp:
        zs = None
    else:
        lengths = [z.shape[-1] for z in zs]
        zs = [
            torch.cat([
                z, torch.zeros(z.shape[:-1] + (max(lengths) - z.shape[-1],))
            ], axis=-1) for z in zs
        ]

        zs = torch.cat(zs, axis=0)

    return ys, zs, ys_orig, xs_hat_orig, shapes


def train_step(
    model,
    ys,
    zs,
    shapes,
    opt,
    dataset,
    device="cuda",
    n_reps=1,
    epoch=0,
    scheduler=None,
    qp=0,
    lambdaRD = 0
):

    if lambdaRD == 0:
        lambdaRD = model.lambdaRD
    print("lambda:", lambdaRD)
    model.train()
    model = model.to(device)
    iterator = tqdm(range(n_reps))
    is_fp = isinstance(model, FactorizedPriorOptimized)
    for _ in iterator:
        losses = []
        rates = []
        psnrs = []
        mses = []
        y_rates = []
        z_rates = []
        for ind, image in enumerate(dataset.X):

            image = image.unsqueeze(0)
            image = image.to(device)
            image_size = np.product(image.shape[2:])

            y = ys[
                ind, :, :np.prod(shapes[ind][0][-2:])
            ].reshape(shapes[ind][0])
            if not is_fp:
                z = zs[ind, :, :np.prod(shapes[ind][1][-2:])].reshape(
                    shapes[ind][1]
                )
                results = model(y, z)
            else:
                results = model(y)
            mse_loss = MSELoss()(
                results["x_hat"],
                image
            ) * 255 ** 2
            y_rate = - torch.log2(
                results["likelihoods"]["y"]
            ).sum()
            if not is_fp:
                z_rate = - torch.log2(
                    results["likelihoods"]["z"]
                ).sum()
            else:
                z_rate = torch.tensor(0)
            loss = mse_loss * lambdaRD + (y_rate + z_rate) / image_size

            opt.zero_grad()
            loss.backward()
            opt.step()

            psnr = 10 * torch.log10(255 ** 2 / mse_loss)
            size = (y_rate + z_rate) / image_size

            losses.append(loss.detach().cpu().numpy())
            rates.append(size.detach().cpu().numpy())
            psnrs.append(psnr.detach().cpu().numpy())
            mses.append(mse_loss.detach().cpu().numpy())
            y_rates.append(y_rate.detach().cpu().numpy() / image_size)
            z_rates.append(z_rate.detach().cpu().numpy() / image_size)
        scheduler.step()

        wandb.log({
            f"QP={qp}/loss": np.mean(losses),
            f"QP={qp}/rate": np.mean(rates),
            f"QP={qp}/psnr train": np.mean(psnrs),
            f"QP={qp}/y rate": np.mean(y_rates),
            f"QP={qp}/z rate": np.mean(z_rates),
            f"QP={qp}/lr1": opt.param_groups[0]["lr"],
            f"QP={qp}/lr2": opt.param_groups[1]["lr"]
        })
        iterator.set_description(
            f"epoch: {epoch+1}, loss: {np.mean(losses):.3f}, "
            f" rate: {np.mean(rates):.4f}"
            f" psnr: {np.mean(psnrs):.2f}"
            f" mse: {np.mean(mses):.4f}"
        )


def validation_step(model, dataset, ys, zs, shapes, ys_orig, xs_hat_orig, get_orig=False):
    is_fp = isinstance(model, FactorizedPriorOptimized)
    model.eval()
    model.cpu()
    ys = ys.cpu()
    if not is_fp:
        zs = zs.cpu()
    losses_orig, rates_orig, psnrs_orig, mses_orig = [], [], [], []
    losses, rates, psnrs, mses = [], [], [], []
    y_sizes_orig, z_sizes_orig, y_sizes, z_sizes = [], [], [], []
    for i, image in enumerate(dataset.X):

        image = image.unsqueeze(0).cpu()
        image_size = np.product(image.shape[2:])

        x_hat_orig = xs_hat_orig[i].cpu()
        y_orig = ys_orig[i]

        # summing the size of the y and z latent and getting the bpp
        size_orig = len(y_orig["strings"][0][0])
        size_orig += 0 if is_fp else len(y_orig["strings"][1][0])
        size_orig = size_orig * 8 / image_size
        y_sizes_orig.append(len(y_orig["strings"][0][0]) * 8 / image_size)
        z_sizes_orig.append(
            len(y_orig["strings"][1][0]) * 8 / image_size if not is_fp else 0
        )

        original_mse_loss = MSELoss()(
            (x_hat_orig * 255).to(torch.uint8).float(),
            (image * 255).to(torch.uint8).float()
        )

        original_loss = original_mse_loss * model.lambdaRD
        original_loss = original_loss + size_orig

        with torch.no_grad():
            y = ys[i, :, :np.prod(shapes[i][0][-2:])].reshape(shapes[i][0])
            if not is_fp:
                z = zs[i, :, :np.prod(shapes[i][1][-2:])].reshape(shapes[i][1])
            if is_fp:
                latent = model.compress(y)
            else:
                latent = model.compress(y, z)
            x_hat = model.decompress(
                latent["strings"],
                latent["shape"]
            )["x_hat"]

        size = (len(latent["strings"][0][0]))
        size += 0 if is_fp else len(latent["strings"][1][0])
        size = size * 8 / image_size
        y_sizes.append(len(latent["strings"][0][0]) * 8 / image_size)
        z_sizes.append(
            0 if is_fp else len(latent["strings"][1][0]) * 8 / image_size
        )

        mse_loss = MSELoss()(
            (x_hat * 255).to(torch.uint8).float(),
            (image * 255).to(torch.uint8).float()
        )

        loss = mse_loss * model.lambdaRD + size

        psnr_orig = 10 * torch.log10(255 ** 2 / original_mse_loss)
        psnr = 10 * torch.log10(255 ** 2 / mse_loss)

        losses.append(loss)
        rates.append(size)
        psnrs.append(psnr)
        mses.append(mse_loss)
        losses_orig.append(original_loss)
        rates_orig.append(size_orig)
        psnrs_orig.append(psnr_orig)
        mses_orig.append(original_mse_loss)

    loss = np.mean(losses)
    original_loss = np.mean(losses_orig)
    size = np.mean(rates)
    size_orig = np.mean(rates_orig)
    psnr = np.mean(psnrs)
    psnr_orig = np.mean(psnrs_orig)
    mse = np.mean(mses)
    mse_orig = np.mean(mses_orig)
    y_size = np.mean(y_sizes)
    y_size_orig = np.mean(y_sizes_orig)
    z_size = np.mean(z_sizes)
    z_size_orig = np.mean(z_sizes_orig)
    delta_loss = loss - original_loss
    delta_size = size - size_orig
    delta_psnr = psnr - psnr_orig
    perc_loss = (delta_loss / original_loss) * 100
    perc_size = (delta_size / size_orig) * 100
    perc_psnr = (delta_psnr / psnr_orig) * 100

    print(
        "\n"
        f"Validation loss: {loss:.6f}, Original Loss: {original_loss:.6f}, "
        f"delta: {delta_loss:.6f}, percentage: {perc_loss:.2f}%\n"
        f"Validation rate: {size:.6f}, Original size: {size_orig:.6f}, "
        f"delta: {delta_size:.6f}, percentage: {perc_size:.2f}%\n"
        f"Validation psnr: {psnr:.6f}, Original psnr: {psnr_orig:.6f},"
        f" delta: {delta_psnr:.6f}, percentage: {perc_psnr:.2f}%\n"
        f"Validation mse: {mse:.6f}, Original mse: {mse_orig:.6f}\n"
        f"y size orig: {y_size_orig:.4f}, y size: {y_size:.4f}\n"
        f"z size orig: {z_size_orig:.4f}, z size: {z_size:.4f}\n"
    )

    if get_orig:
        return losses, rates, psnrs, rates_orig, psnrs_orig, (x_hat * 255).to(torch.uint8), latent
    else:
        return losses, rates, psnrs


def schedule(const):
    def first_piece(epoch): return const ** epoch
    def second_piece(epoch): return (const ** 700) * (0.9997**(epoch - 700))

    def scheduler_func(epoch): return first_piece(epoch) if epoch < 700 \
        else second_piece(epoch)
    return scheduler_func
