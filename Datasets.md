## Datasets
- [KITTI](http://www.cvlibs.net/datasets/kitti/):  [Are we ready for autonomous driving? the KITTI vision benchmark suite](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf) [CVPR 2012]
- [ModelNet](https://modelnet.cs.princeton.edu): [3d shapenets: A deep representation for volumetric shapes](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf) [CVPR 2015]
- [SUN RGB-D](http://rgbd.cs.princeton.edu/challenge.html): [Sun rgb-d: A rgb-d scene understanding benchmark suite](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_SUN_RGB-D_A_2015_CVPR_paper.pdf) [CVPR 2015]
- [ShapeNet Part](https://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html): [A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://www-cs.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf) [SIGGRAPH Asia 2016]
- [ScanNet](http://www.scan-net.org/): [Scannet: Richly-annotated 3d reconstructions of indoor scenes](http://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf) [CVPR 2017 ]
- [S3DIS](http://buildingparser.stanford.edu/dataset.html): [Joint 2D-3D-Semantic Data for Indoor Scene Understanding](https://arxiv.org/pdf/1702.01105.pdf) [arXiv 2017]
- [Semantic3D](http://www.semantic3d.net/): [Semantic3D.net: A new Large-scale Point Cloud Classification Benchmark](https://arxiv.org/pdf/1704.03847.pdf) [arXiv 2017]
- [SemanticKITTI](http://semantic-kitti.org/): [SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences](https://arxiv.org/pdf/1904.01416.pdf) [ICCV 2019]
- [PartNet](https://cs.stanford.edu/~kaichun/partnet/): [PartNet: A Large-scale Benchmark for Fine-grained and
Hierarchical Part-level 3D Object Understanding](https://arxiv.org/pdf/1812.02713.pdf) [CVPR 2019]

<hr />

## 数据集概况(更新中..)
### 一、ModelNet
普林斯顿ModelNet项目的目标是为计算机视觉、计算机图形学、机器人和认知科学领域的研究者们提供一个全面、干净的三维CAD模型集合, 该数据的主页地址[https://modelnet.cs.princeton.edu](https://modelnet.cs.princeton.edu/), 数据最早发布在论文[3D ShapeNets: A Deep Representation for Volumetric Shapes](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf) [CVPR 2015]上。

相关工作人员从数据中选择了常见的40类和10类构成数组子集, 分别表示为ModelNet40和ModelNet10, 且两个数据集都有orientation  aligned的版本。实验中数据用到比较多的是ModelNet40, 有如下三种数据形式:

| 数据集 | modelnet40_normal_resampled.zip | modelnet40_ply_hdf5_2048.zip | ModelNet40.zip |
| :---: | :---: | :---: | :---: |
| 文件大小 | 1.71G | 435M | 2.04G |
| 内容 | point: x, y, z, normal_x, normal_y, normal_z;  <br /> shape: 10k points | point: x, y, z;  <br /> shape: 2048 points | off格式, 具体参考[这里](https://segeval.cs.princeton.edu/public/off_format.html) | 
| 训练集 / 测试集 | 9843 / 2468 | 9840 / 2468 | 9844 / 2468 |
| 下载地址 | [modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) | [modelnet40_ply_hdf5_2048.zip](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) | [ModelNet40.zip](http://modelnet.cs.princeton.edu/ModelNet40.zip) |
