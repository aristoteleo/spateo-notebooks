import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, "../../scripts/methods")
from my_SPACEL import spacel_align
import tracemalloc

import time
from datetime import datetime


results_folder = './results/SPACEL_CPU/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
mem_results_folder = './results/SPACEL_CPU_mem/'
if not os.path.exists(mem_results_folder):
    os.makedirs(mem_results_folder)
slice1 = ad.read_h5ad('slice1_pca.h5ad')
slice2 = ad.read_h5ad('slice2_pca.h5ad')
slice1 = slice1[~slice1.obs['SubClass'].values.isna()]
slice2 = slice2[~slice2.obs['SubClass'].values.isna()]

subsample_nums = [int(i)+1 for i in np.exp(np.log(100) + np.arange(19)*np.log(10) / 5)] + [500000]

for i, subsample_num in enumerate(subsample_nums):
    print(f"-----------{subsample_num}-----------")
    current_time = datetime.now()
    print(f"------Current time: {current_time}------")
    # subsample slices
    subsample1 = np.random.choice(slice1.shape[0], subsample_num, replace=False) if subsample_num < slice1.shape[0] else np.arange(slice1.shape[0])
    subsample2 = np.random.choice(slice2.shape[0], subsample_num, replace=False) if subsample_num < slice2.shape[0] else np.arange(slice2.shape[0])
    sub_slice1 = slice1[subsample1,:].copy()
    sub_slice2 = slice2[subsample2,:].copy()

    
    if i == 0:
        align_slices = spacel_align(
            models=[sub_slice1, sub_slice2],
            spatial_key="spatial",
            key_added="align_spatial",
            anno_key = "SubClass",
        )

    tracemalloc.start()
    time_start = time.time()
    align_slices = spacel_align(
        models=[sub_slice1, sub_slice2],
        spatial_key="spatial",
        key_added="align_spatial",
        anno_key = "SubClass",
    )
    time_end = time.time()
    t = time_end - time_start
    current, peak = tracemalloc.get_traced_memory()
    print(f"当前内存使用: {current / 10**6} MB; 内存使用峰值: {peak / 10**6} MB")

    tracemalloc.stop()
    
    np.save(os.path.join(results_folder, '{}.npy'.format(subsample_num)), t, allow_pickle=True)
    np.save(os.path.join(mem_results_folder, '{}.npy'.format(subsample_num)), peak / 10**6, allow_pickle=True)
