import os
from tempfile import mkstemp
from itertools import product
import multiprocessing as mp
import numpy as np
import torch
from torchvision.io import write_jpeg, read_image
from torchvision.transforms import Resize, CenterCrop
from compressai.transforms.functional import rgb2ycbcr, ycbcr2rgb
from compressai.zoo import cheng2020_anchor, mbt2018, mbt2018_mean, bmshj2018_hyperprior
from compressai.zoo import bmshj2018_factorized
from tqdm import tqdm


def crop_image_to_correct_size(image):
    h = image.shape[1]
    w = image.shape[2]
    new_h = h - h % 64
    new_w = w - w % 64
    return CenterCrop((new_h, new_w))(image)

# get the image listfrom the folder


def images_in_path(path):
    images = [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(path)
        for f in files if f.endswith(".png")
    ]
    return images


def encode_with_jpeg(image, filename, qp):
    write_jpeg(image, filename, quality=qp)
    size = os.stat(filename).st_size
    reconstructed_image = read_image(filename)
    return reconstructed_image, size


def encode_with_compressai(image, qp, constructor):
    model = constructor(qp, metric="mse", pretrained=True, progress=True)
    model.eval()
    initial_shape = image.shape
    X = image / 255
    X = X.unsqueeze(0)
    with torch.no_grad():
        y = model.compress(X)
        x_hat = model.decompress(y["strings"], y["shape"])["x_hat"]
    x_hat = x_hat.cpu()
    x_hat = Resize(initial_shape[1:])(x_hat) * 255
    size = len(y["strings"][0][0]) + len(y["strings"][1][0])
    return x_hat[0].to(torch.uint8), float(size)


def encode_with_factorized_prior(image, qp):
    model = bmshj2018_factorized(
        qp,
        metric="mse",
        pretrained=True,
        progress=True
    )
    model.eval()
    initial_shape = image.shape
    X = image / 255
    X = X.unsqueeze(0)
    with torch.no_grad():
        y = model.compress(X)
        x_hat = model.decompress(y["strings"], y["shape"])["x_hat"]
    x_hat = x_hat.cpu()
    x_hat = Resize(initial_shape[1:])(x_hat) * 255
    size = len(y["strings"][0][0])
    return x_hat[0].to(torch.uint8), float(size)

def encode_with_VVC(
    index,
    filename,
    qp
):
    print(f"Encoding {filename} with qp: {qp}")
    image = read_image(filename)
    h, w = image.shape[1:]
    I = image.float() / 255
    I = np.clip(rgb2ycbcr(I).numpy(), 0, 1)
    I = (I * 255).astype(np.uint8)
    fd, yuv_path = mkstemp(suffix=".yuv")
    out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
    with open(yuv_path, "wb") as f:
        f.write(I.tobytes())

    encoder_path = os.path.join(
        "..",
        "..",
        "VVCSoftware_VTM",
        "bin",
        "EncoderAppStatic"
    )
    decoder_path = os.path.join(
        "..",
        "..",
        "VVCSoftware_VTM",
        "bin",
        "DecoderAppStatic"
    )
    config_path = os.path.join(
        "..",
        "..",
        "VVCSoftware_VTM",
        "cfg",
        "encoder_intra_vtm.cfg"
    )
    cmd = [
        encoder_path,
        f"-i {yuv_path}",
        f"-c {config_path}",
        f"-q {qp}",
        "-o /dev/null",
        f"-b {out_filepath}",
        f"-wdt {w}",
        f"-hgt {h}",
        "-fr 1",
        "-f 1",
        "--InputChromaFormat=444",
        "--InputBitDepth=8",
        "--ConformanceWindowMode=1",
        "> /dev/null 2>&1"
    ]
    os.system(" ".join(cmd))
    os.close(fd)
    os.unlink(yuv_path)
    cmd = [
        decoder_path,
        f"-b {out_filepath}",
        f"-o {yuv_path}",
        "-d 8",
        "> /dev/null 2>&1"
    ]
    os.system(" ".join(cmd))

    rec = torch.tensor(np.fromfile(yuv_path, dtype=np.uint8).reshape((3, h, w)))
    rec = rec.float() / 255
    rec = ycbcr2rgb(rec).clip(0,1) * 255

    size = os.stat(out_filepath).st_size
    os.unlink(yuv_path)
    os.unlink(out_filepath)
    return index, filename, rec, size

def evaluate_methods(images, plot_dict, compute_references):
    for image_path in images:
        if image_path not in plot_dict:
            plot_dict[image_path] = {}

    vvc_range = list(range(47, 12, -5))

    parall_args = [
        (i, image_path, qp) for image_path, (i, qp) in product(images, enumerate(vvc_range))
        if compute_references or "VVC" not in plot_dict[image_path]
    ]

    if len(parall_args) > 0:
        num_jobs = min(os.cpu_count(), len(parall_args))
        with mp.Pool(num_jobs) as pool:
            res = pool.starmap(encode_with_VVC, parall_args)
        for image_path in images:
            if "VVC" not in plot_dict[image_path]:
                plot_dict[image_path]["VVC"] = {}
            plot_dict[image_path]["VVC"]["rate"] = [None for _ in vvc_range]
            plot_dict[image_path]["VVC"]["psnr"] = [None for _ in vvc_range]

        for i, image_path, rec, size in res:
            image = read_image(image_path)
            plot_dict[image_path]["VVC"]["rate"][i] = size

            msevvc = (
                image.float() - rec.float()
            ).square().mean().numpy()
            psnr_value = 10.0 * np.log10(255.0 ** 2 / msevvc)
            plot_dict[image_path]["VVC"]["psnr"][i] = psnr_value

    iterator = tqdm(images)
    for image_path in iterator:
        image = crop_image_to_correct_size(read_image(image_path))
        iterator.set_description(f"Current image: {image_path}")

        if compute_references or "Cheng2020" not in plot_dict[image_path]:
            plot_dict[image_path]["Cheng2020"] = {}
            sizes_cheng = []
            psnrs_cheng = []
            for qp in range(1, 7):
                cheng_image, cheng_size = encode_with_compressai(
                    image,
                    qp,
                    cheng2020_anchor
                )
                msecheng = (
                    image.float() - cheng_image.float()
                ).square().mean().numpy()
                psnrs_cheng.append(10.0 * np.log10(255.0 ** 2 / msecheng))
                sizes_cheng.append(cheng_size)
            plot_dict[image_path]["Cheng2020"]["rate"] = sizes_cheng
            plot_dict[image_path]["Cheng2020"]["psnr"] = psnrs_cheng

        if compute_references or "JAHP" not in plot_dict[image_path]:
            plot_dict[image_path]["JAHP"] = {}
            sizes_JAHP = []
            psnrs_JAHP = []
            for qp in range(1, 9):
                JAHP_image, JAHP_size = encode_with_compressai(
                    image,
                    qp,
                    mbt2018
                )
                mseJAHP = (
                    image.float() - JAHP_image.float()
                ).square().mean().numpy()
                psnrs_JAHP.append(10.0 * np.log10(255.0 ** 2 / mseJAHP))
                sizes_JAHP.append(JAHP_size)
            plot_dict[image_path]["JAHP"]["rate"] = sizes_JAHP
            plot_dict[image_path]["JAHP"]["psnr"] = psnrs_JAHP

        if compute_references or "MSH" not in plot_dict[image_path]:
            plot_dict[image_path]["MSH"] = {}
            sizes_MSH = []
            psnrs_MSH = []
            for qp in range(1, 9):
                MSH_image, MSH_size = encode_with_compressai(
                    image,
                    qp,
                    mbt2018_mean
                )
                mseMSH = (
                    image.float() - MSH_image.float()
                ).square().mean().numpy()
                psnrs_MSH.append(10.0 * np.log10(255.0 ** 2 / mseMSH))
                sizes_MSH.append(MSH_size)
            plot_dict[image_path]["MSH"]["rate"] = sizes_MSH
            plot_dict[image_path]["MSH"]["psnr"] = psnrs_MSH

        if compute_references or "SH" not in plot_dict[image_path]:
            plot_dict[image_path]["SH"] = {}
            sizes_SH = []
            psnrs_SH = []
            for qp in range(1, 9):
                SH_image, SH_size = encode_with_compressai(
                    image,
                    qp,
                    bmshj2018_hyperprior
                )
                mseSH = (
                    image.float() - SH_image.float()
                ).square().mean().numpy()
                psnrs_SH.append(10.0 * np.log10(255.0 ** 2 / mseSH))
                sizes_SH.append(SH_size)
            plot_dict[image_path]["SH"]["rate"] = sizes_SH
            plot_dict[image_path]["SH"]["psnr"] = psnrs_SH

        if compute_references or "FP" not in plot_dict[image_path]:
            plot_dict[image_path]["FP"] = {}
            sizes_FP = []
            psnrs_FP = []
            for qp in range(1, 9):
                FP_image, FP_size = encode_with_factorized_prior(
                    image,
                    qp
                )
                mseFP = (
                    image.float() - FP_image.float()
                ).square().mean().numpy()
                psnrs_FP.append(10.0 * np.log10(255.0 ** 2 / mseFP))
                sizes_FP.append(FP_size)
            plot_dict[image_path]["FP"]["rate"] = sizes_FP
            plot_dict[image_path]["FP"]["psnr"] = psnrs_FP

    return plot_dict
