import os
import numpy as np
import open3d as o3d
import pickle
import random


def vis_npys(npys):
    pcds = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for i, npy in enumerate(npys):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(npy)
        if i < 3:
            color = colors[i]
        else:
            color = [random.random() for _ in range(3)]
        pcd.paint_uniform_color(color)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)


def decode_points(pts_filename):
    '''
    # 3DMatch_train_0.030_points.pkl: dict
    # key: str, sun3d-brown_bm_1-brown_bm_1/seq-01/cloud_bin_0
    # value: np.ndarray n x 3, n是变化的
    # 3933个点云
    # min: (850, 3), max: (197343, 3), mean: 13565
    '''
    with open(pts_filename, 'rb') as file:
        data = pickle.load(file)
        points = [*data.values()]
        ids_list = [*data.keys()]

    dims = []
    for i in range(len(points)):
        dims.append(points[i].shape[0])
    print('npts min: {}, npts max: {}, npts mean: {}'.
          format(min(dims), max(dims), np.mean(dims)))
    print('Total number of point cloud: {}'.format(len(dims)))

    return data


def decode_overlap(overlap_filename):
    '''
    # 3DMatch_train_0.030_overlap.pkl: dict
    # 35297
    # key: str, '7-scenes-pumpkin/seq-07/cloud_bin_11@7-scenes-pumpkin/seq-08/cloud_bin_2'
    # val: float,  0.6015544397826535
    # min: 0.30000815461143276, max: 0.9954887218045113, mean: 0.5150335449363996
    '''
    with open(overlap_filename, 'rb') as file:
        overlap = pickle.load(file)

    scores = []
    for k, v in overlap.items():
        scores.append(v)
    print('overlap min: {}, overlap max: {}, overlap mean: {}'
          .format(min(scores), max(scores), np.mean(scores)))
    print('Total pairs: {}'.format(len(scores)))

    return overlap


def decode_keypts(keypts_filename):
    '''
    # 3DMatch_train_0.030_keypts.pkl: dict
    # 35297
    # key: str, analysis-by-synthesis-office2-5b/seq-01/cloud_bin_34@analysis-by-synthesis-office2-5b/seq-01/cloud_bin_35
    # val: np.ndarray, m x 2; m是变化的
    # min: 445, max: 76307, mean: 8487

    '''
    with open(keypts_filename, 'rb') as file:
        correspondences = pickle.load(file)

    pairs = []
    for k, v in correspondences.items():
        pairs.append(v.shape[0])
    print('min: {}, max: {}, mean: {}'.format(min(pairs), max(pairs), np.mean(pairs)))
    print('Total pairs: {}'.format(len(pairs)))

    return correspondences


if __name__ == '__main__':
    root = '/Users/zhulf/Downloads/data/backup'
    pts_filename = os.path.join(root, f'3DMatch_train_0.030_points.pkl')
    overlap_filename = os.path.join(root, f'3DMatch_train_0.030_overlap.pkl')
    keypts_filename = os.path.join(root, f'3DMatch_train_0.030_keypts.pkl')

    assert os.path.exists(pts_filename)
    print('='*20, '3DMatch_train_0.030_points.pkl', '='*20)
    data = decode_points(pts_filename)
    print('=' * 20, '3DMatch_train_0.030_points.pkl completed', '=' * 20, '\n')

    assert os.path.exists(keypts_filename)
    print('=' * 20, '3DMatch_train_0.030_overlap.pkl', '=' * 20)
    overlap = decode_overlap(overlap_filename)
    print('=' * 20, '3DMatch_train_0.030_overlap.pkl completed', '=' * 20, '\n')

    assert os.path.exists(overlap_filename)
    print('=' * 20, '3DMatch_train_0.030_keypts.pkl', '=' * 20)
    correspondences = decode_keypts(keypts_filename)
    print('=' * 20, '3DMatch_train_0.030_keypts.pkl completed', '=' * 20)


    f1f2 = list(correspondences.keys())[222]
    correspondence = correspondences[f1f2]
    path1, path2 = f1f2.split('@')
    npy1, npy2 = data[path1], data[path2]
    npy3, npy4 = npy1[correspondence[:, 0]], npy2[correspondence[:, 1]]
    vis_npys([npy1, npy2])
    vis_npys([npy1, npy2, npy3])
    vis_npys([npy1, npy2, npy3])