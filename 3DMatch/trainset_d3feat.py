from os.path import join, exists
import pickle
import numpy as np


root = '/Users/zhulf/Downloads/data/backup'

pts_filename = join(root, f'3DMatch_train_0.030_points.pkl')
keypts_filename = join(root, f'3DMatch_train_0.030_keypts.pkl')
overlap_filename = join(root, f'3DMatch_train_0.030_overlap.pkl')


if exists(pts_filename) and exists(keypts_filename):
    with open(pts_filename, 'rb') as file:
        data = pickle.load(file)
        points = [*data.values()]
        ids_list = [*data.keys()]
    with open(keypts_filename, 'rb') as file:
        correspondences = pickle.load(file)

    with open(overlap_filename, 'rb') as file:
        overlap = pickle.load(file)
    print(f"Load PKL file from {pts_filename}")

scores = []
for k, v in overlap.items():
    print(k, v)
    scores.append(v)
print('min: {}, max: {}, mean: {}'.format(min(scores), max(scores), np.mean(scores)))

'''
pairs = []
for k, v in correspondences.items():
    print(k, v)
    print(v.shape)
    pairs.append(v.shape[0])
print('min: {}, max: {}, mean: {}'.format(min(pairs), max(pairs), np.mean(pairs)))
'''

# 3DMatch_train_0.030_points.pkl: dict
# key: str, sun3d-brown_bm_1-brown_bm_1/seq-01/cloud_bin_0
# value: np.ndarray n x 3, n是变化的
# 3933个点云
# min: (850, 3), max: (197343, 3), mean: 13565

# 3DMatch_train_0.030_keypts.pkl: dict
# 35297
# key: str, analysis-by-synthesis-office2-5b/seq-01/cloud_bin_34@analysis-by-synthesis-office2-5b/seq-01/cloud_bin_35
# val: np.ndarray, m x 2; m是变化的
# min: 445, max: 76307, mean: 8487

# 3DMatch_train_0.030_overlap.pkl: dict
# 35297
# key: str, '7-scenes-pumpkin/seq-07/cloud_bin_11@7-scenes-pumpkin/seq-08/cloud_bin_2'
# val: float,  0.6015544397826535
# min: 0.30000815461143276, max: 0.9954887218045113, mean: 0.5150335449363996

'''
dims = []
for i in range(len(points)):
    print(i, points[i].shape)
    dims.append(points[i].shape)

print('min: {}, max: {}, mean: {}'.format(min(dims), max(dims), np.mean(dims)))
'''