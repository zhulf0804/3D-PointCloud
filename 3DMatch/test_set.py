'''
This code is for analysing 3DMatch test set
'''

import copy
import glob
import numpy as np
import os
import open3d as o3d
import random


def basi_info(root):
    '''
    7-scenes-redkitchen: 60
    sun3d-home_at-home_at_scan1_2013_jan_1: 60
    sun3d-home_md-home_md_scan9_2012_sep_30: 60
    sun3d-hotel_uc-scan3: 55
    sun3d-hotel_umd-maryland_hotel1: 57
    sun3d-hotel_umd-maryland_hotel3: 37
    sun3d-mit_76_studyroom-76-1studyroom2: 66
    sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika: 38
    '''
    d = {}
    clss = []
    for item in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, item)):
            clss.append(item)

    for cls in clss:
        if '-evaluation' in cls:
            continue
        plys = glob.glob(os.path.join(root, cls, '*.ply'))
        d[cls] = len(plys)
    return d


def read_gt_log(file_path):
    '''
    7-scenes-redkitchen 506
    sun3d-home_at-home_at_scan1_2013_jan_1 156
    sun3d-home_md-home_md_scan9_2012_sep_30 208
    sun3d-hotel_uc-scan3 226
    sun3d-hotel_umd-maryland_hotel1 104
    sun3d-hotel_umd-maryland_hotel3 54
    sun3d-mit_76_studyroom-76-1studyroom2 292
    sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika 77
    '''
    Rts = []
    with open(file_path, 'r') as fr:
        line = fr.readline()
        while line:
            i, j, _ = list(map(int, line.split()))
            Rt = np.eye(4, dtype=np.float32)
            for t in range(4):
                cur = fr.readline()
                Rt[t, :] = np.fromstring(cur, np.float32, sep='\t')
            Rts.append([[i, j], Rt])
            line = fr.readline()
    return Rts


def vis_gt_log(root, cls, pairs_Rt):
    print(cls, pairs_Rt[0])
    i, j = pairs_Rt[0]
    Rt = pairs_Rt[1]
    ply1_path = os.path.join(root, cls, 'cloud_bin_{}.ply'.format(i))
    ply2_path = os.path.join(root, cls, 'cloud_bin_{}.ply'.format(j))
    ply1 = o3d.io.read_point_cloud(ply1_path, format='ply')
    ply2 = o3d.io.read_point_cloud(ply2_path, format='ply')
    raw_ply2 = copy.deepcopy(ply2)
    ply2.transform(Rt)
    ply1.paint_uniform_color([1, 0, 0])
    raw_ply2.paint_uniform_color([0, 1, 0])
    ply2.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([ply1, ply2])


if __name__ == '__main__':
    root = '/Users/zhulf/data/threedmatch-test'

    name2num = basi_info(root)
    print('=' * 20, 'Plys', '=' * 20)
    for k, v in name2num.items():
        print(k, v)
    print('Total plys numbers: ', sum(name2num.values()))

    print('='*20, 'Correspondences', '='*20)
    name2correspondences = {}
    ncorrespondences = 0
    for k in name2num.keys():
        gt_log_file_path = os.path.join(root, '{}-evaluation'.format(k), 'gt.log')
        Rts = read_gt_log(file_path=gt_log_file_path)
        name2correspondences[k] = Rts
        ncorrespondences += len(Rts)
        print(k, len(Rts))
    print('Total correspondences pairs: ', ncorrespondences)

    #cls = random.choice(list(name2correspondences.keys()))
    #pairs_Rt = random.choice(name2correspondences[cls])
    cls = '7-scenes-redkitchen'
    pairs_Rt = name2correspondences[cls][0]
    vis_gt_log(root, cls, pairs_Rt)