#read in image and apply single noise transforms
import os
import time
import argparse
import torch
import glob
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from util import cal_psnr,cal_ssim
from util import AverageMeter
from ssim_v2 import ssim_v2
from BM3D_py.utils import add_gaussian_noise, symetrize
from BM3D_py.bm3d_1st_step import bm3d_1st_step
from BM3D_py.bm3d_2nd_step import bm3d_2nd_step
from BM3D_py.psnr import compute_psnr


def save_image_single(ground_truth, noisy_image, clean_image, filepath):
    
    ground_image = Image.fromarray(ground_truth.astype('uint8')).convert('L')
    ground_image.save("%s_ground.png"%filepath, 'png')
    noisy_image = Image.fromarray(noisy_image.astype('uint8')).convert('L')
    noisy_image.save("%s_noise.png"%filepath, 'png')
    clean_image = Image.fromarray(clean_image.astype('uint8')).convert('L')
    clean_image.save("%s_clean.png"%filepath, 'png')

def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised

to_tensor = transforms.ToTensor()
class SingleTransform(object):
    def __init__(self, noise_level, interpolation=Image.BILINEAR,noise_type = 1):
        self.interpolation = interpolation
        self.noise_level = noise_level
        self.noise_type = noise_type

    def __call__(self, image):
        batch_x = to_tensor(image)
        noise_type = self.noise_type
        #noise_type = random.randint(1,3)
        thre = self.noise_level/100
        #thre = (random.randint(1,80))/100
        thre_a = thre
        if noise_type == 1:  #gaussian
            #rand_sigma = 25#random.randint(0,50)
            rand_sigma = self.noise_level
            noise = torch.randn(batch_x.size()).mul_(rand_sigma/255.0)
            batch_y = batch_x + noise
            thre_a = rand_sigma/255.0
        elif noise_type == 2:  #impluse1                 
            noise_gt = torch.rand(batch_x.size())
            noise = noise_gt.clone()     
            noise_image = batch_x.clone()
            noise[noise>=thre] = 0
            noise_tmp = noise[noise>0]
            lmn = torch.min(noise_tmp)
            lmax = torch.max(noise_tmp)
            noise = ((noise-lmn))/(lmax-lmn)
            mask = torch.ceil(noise_gt-thre)
            noise_image = noise*(1-mask)+noise_image*mask
            batch_y = noise_image
        
        elif noise_type == 3: #impluse2(salt-and-pepper)
            thre = thre/2
            noise_gt = torch.rand(batch_x.size())
            noise = noise_gt.clone()     
            noise_image = batch_x.clone()
        
            noise_image[noise<thre]=0
            noise_image[(noise>thre)&(noise<2*thre)]=1
            batch_y = noise_image
        elif noise_type == 4: #poisson
            p = torch.distributions.poisson.Poisson(thre*255*batch_x)
            batch_y = p.sample()/(255*thre)
        elif noise_type == 5: #multi
            rand_sigma = 25#random.randint(0,50)
            noise = torch.randn(batch_x.size()).mul_(rand_sigma/255.0)
            batch_y = batch_x + noise*batch_x
            thre_a = rand_sigma/255.0
           
        noise_image = torch.clamp(255 * batch_y, 0, 255).byte().numpy()
        im_c,im_h,im_w = noise_image.shape
        noise_image = noise_image.reshape((im_h, im_w, im_c)) 
        #noise_image = (batch_y.permute(1,2,0)*255).numpy().astype(np.uint8)
        
        return noise_image

if __name__ == "__main__":
    print("============> Statrt!")    
    parser = argparse.ArgumentParser(description="PyTorch denoise Experiment")
    parser.add_argument('--result_dir', dest='result_dir', default='./', help='Test image result dir')
    parser.add_argument('--test_dir', dest='test_dir', default='./', help='Test data dir')
    parser.add_argument('--noise_level', type=int, default=50, help='Noise level')
    parser.add_argument("--n_type", default=1, type=int, help="noise type 1-Gaussian 2-random impulse 3-salt-and-pepper 4-Poisson")

    # <hyper parameter> -------------------------------------------------------------------------------
    n_H = 16
    k_H = 8
    N_H = 16
    p_H = 3
    lambda3D_H = 2.7  # ! Threshold for Hard Thresholding
    useSD_H = False
    tau_2D_H = 'BIOR'

    n_W = 16
    k_W = 8
    N_W = 32
    p_W = 3
    useSD_W = True
    tau_2D_W = 'DCT'
    # <\ hyper parameter> -----------------------------------------------------------------------------

    batch_time = AverageMeter()
    avg_psnr = AverageMeter()
    avg_ssim = AverageMeter()

    global opt
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    if not os.path.exists(opt.test_dir):
        print("{} not exist".format(opt.test_dir))
        exit()

    tauMatch_H = 2500 if opt.noise_level < 35 else 5000  # ! threshold determinates similarity between patches
    tauMatch_W = 400 if opt.noise_level < 35 else 3500  # ! threshold determinates similarity between patches

    test_transform = SingleTransform(noise_level = opt.noise_level,noise_type = opt.n_type)

    image_filenames = glob.glob('{}/*.png'.format(opt.test_dir))

    value_out_path = os.path.join(opt.result_dir, "eval_result.txt")
    F = open(value_out_path,'a')

    for i in range(len(image_filenames)):
        image = Image.open(image_filenames[i]).convert('L')
        noise_image = test_transform(image)
        if noise_image.ndim == 3:
            h,w,c = noise_image.shape
            noise_image.resize(h,w)
        clean_image = np.array(image)
        end = time.time()
        
        m1, im2 = run_bm3d(noise_image, opt.noise_level,
                           n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                           n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

        batch_time.update(time.time() - end)
        
        noise_image = (np.clip(noise_image, 0, 255)).astype(np.uint8)
        clean_image = (np.clip(clean_image, 0, 255)).astype(np.uint8)
        denoise_image = (np.clip(im2, 0, 255)).astype(np.uint8)
        
        save_image_single(clean_image, noise_image, denoise_image, os.path.join(opt.result_dir, 'test%d.png'%i))
        psnr = cal_psnr(clean_image, denoise_image)
        ssim = ssim_v2(clean_image, denoise_image)

        avg_psnr.update(psnr)
        avg_ssim.update(ssim)

        print('ProcessImage NO[{0}/{1}]:\t'
          'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
          'Ssim {ssim.val:.4f} ({ssim.avg:.4f})\t'
          'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
           i, len(image_filenames),batch_time=batch_time, ssim=avg_ssim, psnr=avg_psnr))

        F.write('Image NO[{0}/{1}]:\t'
          'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
          'Ssim {ssim.val:.4f} ({ssim.avg:.4f})\t'
          'Psnr {psnr.val:.3f} ({psnr.avg:.3f})\n'.format(
           i, len(image_filenames),batch_time=batch_time, ssim=avg_ssim, psnr=avg_psnr))
    
    print("--- FINISH! --------- Average PSNR %.3f ---" %(avg_psnr.avg))

    F.write("Dataset Average: PSNR %.3f ssim %.4f\n"%(avg_psnr.avg,avg_ssim.avg))
    F.close()

