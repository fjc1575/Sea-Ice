import os

from osgeo import gdal_array
import numpy as np
import get_glcm
import time
from PIL import Image

def main():
    pass

def GLCM_Features(inPath, outPath):

    start = time.time()

    # ('---------------0. Parameter Setting-----------------')
    nbit = 32 # gray levels 灰度级数
    mi, ma = 0, 256 # max gray and min gray 初始的最小和最大灰度值数
    slide_window = 5 # sliding window 滑动窗口的大小
    # step = [2, 4, 8, 16] # step
    angle = [0, np.pi/4, np.pi/2, np.pi*3/4] # angle or direction 方向 0 45° 90° 135°
    step = [1]  # 步长
    # angle = [0]  # 方向
    # ('-------------------1. Load Data---------------------')
    # image = r"./test.tif"
    image = inPath
    img = np.array(Image.open(image)) # If the image has multi-bands, it needs to be converted to grayscale image
    img = np.uint8(255.0 * (img - np.min(img))/(np.max(img) - np.min(img)))  # normalization 归一化，将影像拉伸到0-255范围内
    h, w = img.shape  # 获取影像大大小
    # print('------------------2. Calcu GLCM---------------------')
    glcm = get_glcm.calcu_glcm(img, mi, ma, nbit, slide_window, step, angle)  # 计算灰度共生矩阵
    # ('-----------------3. Calcu Feature-------------------')
    #
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = np.zeros((nbit, nbit, h, w), dtype=np.float32)
            glcm_cut = glcm[:, :, i, j, :, :]
            mean = get_glcm.calcu_glcm_mean(glcm_cut, nbit)  # 返回的mean的大小是512*512，与原图一致
            # variance = get_glcm.calcu_glcm_variance(glcm_cut, nbit)
            # homogeneity = get_glcm.calcu_glcm_homogeneity(glcm_cut, nbit)
            # contrast = get_glcm.calcu_glcm_contrast(glcm_cut, nbit)
            # dissimilarity = get_glcm.calcu_glcm_dissimilarity(glcm_cut, nbit)
            # entropy = get_glcm.calcu_glcm_entropy(glcm_cut, nbit)
            # energy = get_glcm.calcu_glcm_energy(glcm_cut, nbit)
            # correlation = get_glcm.calcu_glcm_correlation(glcm_cut, nbit)
            Auto_correlation = get_glcm.calcu_glcm_Auto_correlation(glcm_cut, nbit)
    # ('---------------4. Save and Result----------------')


    # 转化成ndarray形式
    img_np = np.array(img)
    B1_np = np.array(mean)
    # B2_np = np.array(variance)
    # B3_np = np.array(homogeneity)
    # B4_np = np.array(contrast)
    # B5_np = np.array(dissimilarity)
    # B6_np = np.array(entropy)
    # B7_np = np.array(energy)
    # B8_np = np.array(correlation)
    B9_np = np.array(Auto_correlation)

    img = np.stack([img_np, B1_np, B9_np], axis=0)

    # 按照通道堆叠

    path = outPath
    gdal_array.SaveArray(img, path, format="GTiff")

    end = time.time()
    print('Code run time:', end - start)


