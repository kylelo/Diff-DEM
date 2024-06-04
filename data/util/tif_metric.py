import os
import numpy as np
import cv2
import torch
from piq import multi_scale_ssim, psnr
from skimage.transform import resize


def crop_to_mask(image, mask):
    # Find the bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the image to the bounding box
    return image[rmin:rmax+1, cmin:cmax+1]

def ms_ssim(gt_image, image, mask):
    gt_tensor = torch.tensor(np.array(gt_image))
    pred_tensor = torch.tensor(np.array(image))
    
    mask = mask > 0

    gt_tensor = crop_to_mask(gt_tensor, mask)
    pred_tensor = crop_to_mask(pred_tensor, mask)

    gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)
    pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)

    max_val = max(torch.max(gt_tensor).item(), torch.max(pred_tensor).item())

    gt_tensor[gt_tensor<0] = 0
    pred_tensor[pred_tensor<0] = 0

    scale_weights = torch.tensor([0.0448, 0.2856, 0.3001])
    ms_ssim_val = multi_scale_ssim(gt_tensor, pred_tensor, data_range=max_val, scale_weights=scale_weights)

    return ms_ssim_val.item()

def calc_psnr(gt_image, pred_image, mask):
    gt_tensor = torch.tensor(np.array(gt_image))
    pred_tensor = torch.tensor(np.array(pred_image))

    mask = mask > 0

    gt_tensor = crop_to_mask(gt_tensor, mask)
    pred_tensor = crop_to_mask(pred_tensor, mask)

    gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)
    pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)

    max_val = max(torch.max(gt_tensor).item(), torch.max(pred_tensor).item())

    gt_tensor[gt_tensor<0] = 0
    pred_tensor[pred_tensor<0] = 0

    psnr_val = psnr(gt_tensor, pred_tensor, data_range=max_val)

    return psnr_val.item()

def compute_rmse(gt_image, image, mask):
    diff = gt_image[mask > 0] - image[mask > 0]
    mse = np.mean(np.square(diff))
    rmse = np.sqrt(mse)
    return rmse

def compute_mae(gt_image, image, mask):
    diff = gt_image[mask > 0] - image[mask > 0]
    mae = np.mean(np.abs(diff))
    return mae

def main(gt_tif_dir, algo_dir, mask_dir, normalize):
    rmse_total = []
    mae_total = []
    msssim_total = []
    psnr_total = []
    for gt_filename in os.listdir(gt_tif_dir):
        if gt_filename.endswith(".tif"):
            gt_path = os.path.join(gt_tif_dir, gt_filename)
            algo_path = os.path.join(algo_dir, gt_filename)
            mask_path = os.path.join(mask_dir, gt_filename.replace('.tif', '_mask.png'))

            if os.path.exists(algo_path) and os.path.exists(mask_path):
                gt_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                image = cv2.imread(algo_path, cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                gt_image = gt_image.astype(np.double)
                image = image.astype(np.double)
                mask = mask.astype(np.double)

                if normalize: # Only for Diff-DEM!!
                    cond_min, cond_max = np.min(gt_image), np.max(gt_image)
                    image /= 256
                    image *= (cond_max - cond_min)
                    image += cond_min
                
                if image.shape != (256,256):
                    image = resize(image, gt_image.shape, mode='reflect', anti_aliasing=False)

                rmse = compute_rmse(gt_image, image, mask)
                mae = compute_mae(gt_image, image, mask)
                msssim = ms_ssim(gt_image, image, mask)
                psnr_val = calc_psnr(gt_image, image, mask)

                rmse_total.append(rmse)
                mae_total.append(mae)
                msssim_total.append(msssim)
                psnr_total.append(psnr_val)
            else:
                print(f"Missing files for {gt_filename}")
    
    print(f"MAE: {np.mean(mae_total):.3f} RMSE: {np.mean(rmse_total):.3f} MS-SSIM: {np.mean(msssim_total):.3f} PSNR: {np.mean(psnr_total):.3f}")
    # print(f"{np.mean(mae_total):.3f} & {np.mean(rmse_total):.3f} & {np.mean(msssim_total):.3f} & {np.mean(psnr_total):.3f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate inpainted pairs.")

    # WARNING: Diff-DEM should use normalize!

    # Table I
    parser.add_argument('--gt_tif_dir', type=str, default='./experiments/gavriil/gt', help='Directory of ground truth TIFF files')
    parser.add_argument('--mask_dir', type=str, default='./experiments/gavriil/mask', help='Directory of mask TIFF files')

    parser.add_argument('--algo_dir', type=str, default='./experiments/gavriil/diff', help='Directory of inpainted TIFF files')
    parser.add_argument('--normalize', default=True, action='store_true') 

    # parser.add_argument('--algo_dir', type=str, default='./experiments/gavriil/void_fill', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=False, action='store_true')

    # parser.add_argument('--algo_dir', type=str, default='./experiments/gavriil/kriging', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=False, action='store_true')
    
    # parser.add_argument('--algo_dir', type=str, default='./experiments/gavriil/spline', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=False, action='store_true')

    # parser.add_argument('--algo_dir', type=str, default='./experiments/gavriil/pm_diff', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=False, action='store_true')

    # parser.add_argument('--algo_dir', type=str, default='./experiments/gavriil/generative_model', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=False, action='store_true')
    



    # # Table II
    # parser.add_argument('--gt_tif_dir', type=str, default='./dataset/norway_dem/benchmark/gt', help='Directory of ground truth TIFF files')
    
    # parser.add_argument('--mask_dir', type=str, default='./dataset/norway_dem/benchmark/mask/64-96', help='Directory of mask TIFF files')
    # parser.add_argument('--algo_dir', type=str, default='./experiments/results/Diff-DEM/64-96', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=True, action='store_true')

    # parser.add_argument('--mask_dir', type=str, default='./dataset/norway_dem/benchmark/mask/96-128', help='Directory of mask TIFF files')
    # parser.add_argument('--algo_dir', type=str, default='./experiments/results/Diff-DEM/96-128', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=True, action='store_true')

    # parser.add_argument('--mask_dir', type=str, default='./dataset/norway_dem/benchmark/mask/128-160', help='Directory of mask TIFF files')
    # parser.add_argument('--algo_dir', type=str, default='./experiments/results/Diff-DEM/128-160', help='Directory of inpainted TIFF files')
    # parser.add_argument('--normalize', default=True, action='store_true')

    args = parser.parse_args()
    main(args.gt_tif_dir, args.algo_dir, args.mask_dir, args.normalize)