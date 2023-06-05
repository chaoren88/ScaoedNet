import sys
sys.path.append('./')

import random, os, glob
import numpy as np
from scipy.io import loadmat, savemat
from skimage import io, img_as_ubyte
from ScaoedNet import DN
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


num_pch = 100


model_list = ['./model/JPEG_Nam_41.41_0.9762.pth']

mat_path = './dataset/NAM_patch.mat'

save_denoised_image = False

with torch.no_grad():
    for ii, model_path in enumerate(model_list):
        checkpoint = torch.load(model_path)
        model = DN().cuda()
        # model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
        model.eval()

        print('model: {:02d}, path: {:s}'.format(ii+1, os.path.split(model_path)[1]))
        data_dict = loadmat(mat_path)
        noisy_img = data_dict['noisy_pchs']
        gt_img = data_dict['gt_pchs']
        inputs = np.transpose(noisy_img, axes=[0, 3, 1, 2]).astype('float32') / 255.0
        iter_pch = inputs.shape[0]
        inputs = torch.from_numpy(inputs).cuda()
        psnr = 0
        ssim = 0
        for i in range(int(iter_pch)):
            input = inputs[i,:,:,:].unsqueeze(0)

            _, test_out = model(input)

            im_denoise = test_out.data.cpu()
            im_denoise.clamp_(0.0, 1.0)

            im_denoise = img_as_ubyte(im_denoise[0, ...].numpy().transpose([1, 2, 0]))
            psnr_iter = compare_psnr(im_denoise, gt_img[i,:,:,:], data_range=255)
            ssim_iter = compare_ssim(im_denoise, gt_img[i,:,:,:], data_range=255, gaussian_weights=True, use_sample_covariance=False,
                                     multichannel=True)

            if save_denoised_image:
                io.imsave('./results/nam/' + '{:0>3}_noisy.png'.format(i + 1,), noisy_img[i,:,:,:])
                io.imsave('./results/nam/' + '{:0>3}_{:4.2f}_{:5.4f}.png'.format(i + 1,
                    psnr_iter, ssim_iter), im_denoise)

            psnr += psnr_iter
            ssim += ssim_iter
        psnr = psnr / iter_pch
        ssim = ssim / iter_pch
        print('PSNR: {:4.4f}, SSIM: {:5.4f}'.format(psnr, ssim))
        print("///////////////////////////////////////")



