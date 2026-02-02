# evaluate the restored images with IQA
# PSNR, SSIM, LPIPS are given as example, you can add more IQA in this file
import cv2
import argparse, os, sys, glob
import logging
from datetime import datetime

import os
os.chdir(os.path.dirname(__file__))
import pyiqa

from torch.utils import data as data
import glob
import numpy as np
import math
import random
import torch

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k in opt:
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """from BasicSR
    Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--img_path",
		nargs="+",
		help="path to the input image",
		#default='/home/tiger/gh/dataset/results/Real_Deg/seeSR/cufed/sample00',
		default='/home/tiger/gh/dataset/results/Real_Deg/SUPIR/wr-sr',
	)


	parser.add_argument(
		"--gt_path",
		nargs="+",
		help="path to the gt image, you need to add the paths of gt folders corresponding to init-imgs",
		default='/home/tiger/gh/dataset/WR-SR-testset/Real_Deg/HR'
	)
	
	parser.add_argument(
		"--log",
		type=str,
		nargs="?",
		help="path to the log",
		default='./iqa_results')

	parser.add_argument(
		"--log-name",
		type=str,
		nargs="?",
		help="name of your log",
		default='test',
	)

	parser.add_argument(
		"--num_img",
		type=int,
		nargs="?",
		help="the number of images evaluated in the folder; 0: all the images are evaludated.",
		default=0,
	)

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	os.makedirs(opt.log, exist_ok=True)
	# init logger
	setup_logger('base', opt.log, 'test_' + opt.log_name, level=logging.INFO,
                  screen=True, tofile=True)
	logger = logging.getLogger('base')
	logger.info(opt)
 
	# init metrics: you can add more metrics here

	iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
	iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
	iqa_lpips = pyiqa.create_metric('lpips', device=device)
	iqa_niqe =   pyiqa.create_metric('niqe', device=device)
	iqa_musiq = pyiqa.create_metric('musiq',device=device)
	iqa_clipiqa = pyiqa.create_metric('clipiqa',device=device)
	iqa_fid   = pyiqa.create_metric('fid',device=device)

	# record metrics
	metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'niqe': [], 'fid': [], 'musiq': [], 'clipiqa': []}

	for img_name in sorted(os.listdir(opt.img_path[0])):
		input_sr_path = os.path.join(opt.img_path[0],img_name)
		#input_gt_path = os.path.join(opt.gt_path,img_name).replace('SR','HR').replace('LR','HR') # TODO need to be modified
		input_gt_path = os.path.join(opt.gt_path[0],img_name)

		input_sr_img = cv2.imread(input_sr_path, cv2.IMREAD_COLOR)
		sr = img2tensor(input_sr_img, bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

		input_gt_img = cv2.imread(input_gt_path, cv2.IMREAD_COLOR)
		hr = img2tensor(input_gt_img, bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

		if sr.shape != hr.shape:
			raise NotImplementedError

		# PSNR: convert the ycbcr to calculate
		hr = hr[..., 4:-4, 4:-4] / 255.
		sr = sr[..., 4:-4, 4:-4] / 255.

		PSNR_now = iqa_psnr(sr, hr).item()
		metrics['psnr'].append(PSNR_now)

		ssim_now = iqa_ssim(sr, hr).item()
		metrics['ssim'].append(ssim_now)

		lpips_now = iqa_lpips(sr, hr).item()
		metrics['lpips'].append(lpips_now)

		niqe_now = iqa_niqe(sr).item()
		metrics['niqe'].append(niqe_now)

		musiq_now = iqa_musiq(sr).item()
		metrics['musiq'].append(musiq_now)

		clipiqa_now = iqa_clipiqa(sr).item()
		metrics['clipiqa'].append(clipiqa_now)


	fid_now = iqa_fid(opt.img_path[0],opt.gt_path[0])
	metrics['fid'].append(fid_now)

	for key,value in metrics.items():
		logger.info('{}:{:.6f}'.format(key,sum(value)/len(value)))
		


if __name__ == '__main__':
	main()