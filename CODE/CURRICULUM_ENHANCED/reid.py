# -*- coding: utf-8 -*-
"""
Test curriculum_clustering via a subset of the WebVision dataset v1.0

You can find more information about the dataset here:
https://www.vision.ee.ethz.ch/webvision/2017/

The testing dataset contains extracted features and labels for the first 10 classes of the WebVision dataset 1.0

The class names are local to this repository, but since the features are a large file, it has been made available here:
https://sai-pub.s3.cn-north-1.amazonaws.com.cn/malong-research/curriculumnet/webvision_cls0-9.npy

The test will download the file automatically if it is not available at test-data/input/webvision_cls0-9.npy

"""

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import os
import shutil
import tempfile
import urllib
import numpy as np
from curriculum_clustering import CurriculumClustering
import scipy.io

def test_curriculum_cluster():
    X, y, metadata = load_webvision_data()
    cc = CurriculumClustering(n_subsets=3, verbose=True, random_state=0)
    cc.fit(X, y)
    verify_webvision_expected_clusters(labels=cc.output_labels, n_subsets=cc.n_subsets, metadata=metadata)


def load_webvision_data():
    # X: features
    features_file = 'tests/test-data/input/feat/with_pretrain_imagenet.mat' # 载入ResNet50跑出的特征
    result = scipy.io.loadmat(features_file)
    X = result['feature']

    # y: labels
    cluster_list = 'tests/test-data/input/dataset.txt'  # 列出文件所在路径与对应标签
    with open(cluster_list) as f:
        metadata = [x.strip().split(' ') for x in f]
    y = [int(item[1]) for item in metadata]

    return X, y, metadata


def verify_webvision_expected_clusters(labels, n_subsets, metadata):
    test_dir = 'tests/test-data/output-reid'
    clustered_by_levels = [list() for _ in range(n_subsets+1)]  # 四个空的数组，少的无法分清的单独算（也视为难样本）

    for idx, _ in enumerate(metadata):
        clustered_by_levels[labels[idx]].append(idx)    # 按聚类标签将训练集分为三组
    for idx, level_output in enumerate(clustered_by_levels):
        with open("{}/{}.txt".format(test_dir, idx + 1), 'w') as f:
            for i in level_output:
                f.write("{}\n".format(str.join(' ', metadata[i])))  # 将聚类信息保存到txt

    print("Test is successful.")

if __name__ == "__main__":
    test_curriculum_cluster()
