import copy
import numpy as np
import os
import open3d as o3d
import random


def read_ids(path):
    ids = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ids.append(line.strip())
    return ids


def read_correspondence_pairs(path):
    pairs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pairs.append(line.split())
    return pairs


def vis_pair(path1, path2):
    pc1, pc2 = np.load(path1), np.load(path2)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1['pcd'])
    pcd1_uniform_color = copy.deepcopy(pcd1)
    pcd1_uniform_color.paint_uniform_color([1, 0, 0])
    pcd1.colors = o3d.utility.Vector3dVector(pc1['color'])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2['pcd'])
    pcd2_uniform_color = copy.deepcopy(pcd2)
    pcd2_uniform_color.paint_uniform_color([0, 0, 1])
    pcd2.colors = o3d.utility.Vector3dVector(pc2['color'])
    o3d.visualization.draw_geometries([pcd1_uniform_color, pcd2_uniform_color])
    o3d.visualization.draw_geometries([pcd1, pcd2])


if __name__ == '__main__':
    root = '/Users/zhulf/data/threedmatch'
    train_ids = read_ids('./train.txt')
    files = sorted(os.listdir(root))
    suffixes, clss = {}, {}
    for file in files:
        suffix = file.split('.')[-1]
        suffixes[suffix] = suffixes.get(suffix, 0) + 1
        cls = file.split('@')[0]
        if cls not in train_ids:
            continue

        if '0.30' in file:
            seq_id = file.split('@')[1][:6]
            pairs = read_correspondence_pairs(os.path.join(root, file))
            clss[(cls, seq_id)] = clss.get((cls, seq_id), 0) + len(pairs)
    print(suffixes)
    for cls, num in clss.items():
        print(cls, num)
    ids = set([cls[0] for cls, num in clss.items()])
    print('Total class: {}, npairs: {}'.format(len(ids), sum(clss.values())))

    #id, seq = random.choice(list(clss.keys()))
    id, seq = '7-scenes-chess', 'seq-01'
    id_seq_path = os.path.join(root, '{}@{}-0.30.txt'.format(id, seq))
    pairs = read_correspondence_pairs(id_seq_path)
    #pair = random.choice(pairs)
    pair = pairs[0]
    path1, path2 = os.path.join(root, pair[0]), os.path.join(root, pair[1])
    print(path1, path2)
    vis_pair(path1, path2)

'''
/Users/zhulf/anaconda3/bin/python3.7 /Users/zhulf/data/3D-PointCloud/3DMatch/trainset_fcgf.py
{'txt': 401, 'npz': 2189}
('7-scenes-chess', 'seq-01') 33
('7-scenes-chess', 'seq-02') 36
('7-scenes-chess', 'seq-03') 36
('7-scenes-chess', 'seq-04') 36
('7-scenes-chess', 'seq-05') 36
('7-scenes-chess', 'seq-06') 36
('7-scenes-fire', 'seq-01') 36
('7-scenes-fire', 'seq-02') 36
('7-scenes-fire', 'seq-03') 35
('7-scenes-fire', 'seq-04') 35
('7-scenes-heads', 'seq-01') 31
('7-scenes-heads', 'seq-02') 24
('7-scenes-office', 'seq-01') 26
('7-scenes-office', 'seq-02') 21
('7-scenes-office', 'seq-03') 21
('7-scenes-office', 'seq-04') 17
('7-scenes-office', 'seq-05') 17
('7-scenes-office', 'seq-06') 18
('7-scenes-office', 'seq-07') 26
('7-scenes-office', 'seq-08') 27
('7-scenes-office', 'seq-09') 22
('7-scenes-office', 'seq-10') 31
('7-scenes-pumpkin', 'seq-01') 28
('7-scenes-pumpkin', 'seq-02') 35
('7-scenes-pumpkin', 'seq-03') 33
('7-scenes-pumpkin', 'seq-06') 29
('7-scenes-pumpkin', 'seq-07') 36
('7-scenes-pumpkin', 'seq-08') 36
('7-scenes-stairs', 'seq-01') 4
('7-scenes-stairs', 'seq-02') 6
('7-scenes-stairs', 'seq-03') 6
('7-scenes-stairs', 'seq-04') 6
('7-scenes-stairs', 'seq-05') 4
('7-scenes-stairs', 'seq-06') 6
('analysis-by-synthesis-apt1-kitchen', 'seq-01') 35
('analysis-by-synthesis-apt1-living', 'seq-01') 67
('analysis-by-synthesis-apt2-bed', 'seq-01') 45
('analysis-by-synthesis-apt2-kitchen', 'seq-01') 32
('analysis-by-synthesis-apt2-living', 'seq-01') 30
('analysis-by-synthesis-apt2-luke', 'seq-01') 84
('analysis-by-synthesis-office2-5a', 'seq-01') 38
('analysis-by-synthesis-office2-5b', 'seq-01') 74
('bundlefusion-apt0', 'seq-01') 699
('bundlefusion-apt1', 'seq-01') 851
('bundlefusion-apt2', 'seq-01') 219
('bundlefusion-copyroom', 'seq-01') 301
('bundlefusion-office0', 'seq-01') 486
('bundlefusion-office1', 'seq-01') 477
('bundlefusion-office2', 'seq-01') 139
('bundlefusion-office3', 'seq-01') 224
('rgbd-scenes-v2-scene_01', 'seq-01') 15
('rgbd-scenes-v2-scene_02', 'seq-01') 14
('rgbd-scenes-v2-scene_03', 'seq-01') 14
('rgbd-scenes-v2-scene_04', 'seq-01') 15
('rgbd-scenes-v2-scene_05', 'seq-01') 47
('rgbd-scenes-v2-scene_06', 'seq-01') 40
('rgbd-scenes-v2-scene_07', 'seq-01') 31
('rgbd-scenes-v2-scene_08', 'seq-01') 27
('rgbd-scenes-v2-scene_09', 'seq-01') 10
('rgbd-scenes-v2-scene_10', 'seq-01') 12
('rgbd-scenes-v2-scene_11', 'seq-01') 8
('rgbd-scenes-v2-scene_12', 'seq-01') 11
('rgbd-scenes-v2-scene_13', 'seq-01') 6
('rgbd-scenes-v2-scene_14', 'seq-01') 15
('sun3d-brown_bm_1-brown_bm_1', 'seq-01') 115
('sun3d-brown_bm_4-brown_bm_4', 'seq-01') 48
('sun3d-brown_cogsci_1-brown_cogsci_1', 'seq-01') 64
('sun3d-brown_cs_2-brown_cs2', 'seq-01') 122
('sun3d-brown_cs_3-brown_cs3', 'seq-01') 57
('sun3d-harvard_c11-hv_c11_2', 'seq-01') 10
('sun3d-harvard_c3-hv_c3_1', 'seq-01') 47
('sun3d-harvard_c5-hv_c5_1', 'seq-01') 115
('sun3d-harvard_c6-hv_c6_1', 'seq-01') 38
('sun3d-harvard_c8-hv_c8_3', 'seq-01') 20
('sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika', 'seq-01') 389
('sun3d-hotel_nips2012-nips_4', 'seq-01') 270
('sun3d-hotel_sf-scan1', 'seq-01') 519
('sun3d-mit_32_d507-d507_2', 'seq-01') 301
('sun3d-mit_46_ted_lab1-ted_lab_2', 'seq-01') 285
('sun3d-mit_76_417-76-417b', 'seq-01') 185
('sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika', 'seq-01') 121
('sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika', 'seq-01') 323
Total class: 54, npairs: 7960
'''