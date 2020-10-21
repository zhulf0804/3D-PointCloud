## Datasets
- [KITTI](http://www.cvlibs.net/datasets/kitti/):  [Are we ready for autonomous driving? the KITTI vision benchmark suite](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf) [CVPR 2012]
- [ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html): [A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM](http://mural.maynoothuniversity.ie/8309/1/JM-Benchmark-2014.pdf) [ICRA 2014]
- [Augmented ICL-NUIM](http://redwood-data.org/indoor/dataset.html): [Robust Reconstruction of Indoor Scenes](https://www.researchgate.net/profile/Vladlen_Koltun/publication/279751165_Robust_Reconstruction_of_Indoor_Scenes/links/5599867708ae5d8f393633dc/Robust-Reconstruction-of-Indoor-Scenes.pdf) [CVPR 2015]
- [ModelNet](https://modelnet.cs.princeton.edu): [3d shapenets: A deep representation for volumetric shapes](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf) [CVPR 2015]
- [SUN RGB-D](http://rgbd.cs.princeton.edu/challenge.html): [Sun rgb-d: A rgb-d scene understanding benchmark suite](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_SUN_RGB-D_A_2015_CVPR_paper.pdf) [CVPR 2015]
- [ShapeNet Part](https://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html): [A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://www-cs.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf) [SIGGRAPH Asia 2016]
- [3DMatch](http://3dmatch.cs.princeton.edu/): [3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions](https://arxiv.org/pdf/1603.08182.pdf) [CVPR 2017]
- [ScanNet](http://www.scan-net.org/): [Scannet: Richly-annotated 3d reconstructions of indoor scenes](http://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf) [CVPR 2017 ]
- [S3DIS](http://buildingparser.stanford.edu/dataset.html): [Joint 2D-3D-Semantic Data for Indoor Scene Understanding](https://arxiv.org/pdf/1702.01105.pdf) [arXiv 2017]
- [Semantic3D](http://www.semantic3d.net/): [Semantic3D.net: A new Large-scale Point Cloud Classification Benchmark](https://arxiv.org/pdf/1704.03847.pdf) [arXiv 2017]
- [SemanticKITTI](http://semantic-kitti.org/): [SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences](https://arxiv.org/pdf/1904.01416.pdf) [ICCV 2019]
- [PartNet](https://cs.stanford.edu/~kaichun/partnet/): [PartNet: A Large-scale Benchmark for Fine-grained and
Hierarchical Part-level 3D Object Understanding](https://arxiv.org/pdf/1812.02713.pdf) [CVPR 2019]
- [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset): [Scalability in Perception for Autonomous Driving: Waymo Open Dataset](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf) [CVPR 2020]
- [SensatUrban](https://github.com/QingyongHu/SensatUrban): [Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges](https://arxiv.org/pdf/2009.03137.pdf) [arXiv 2020]

<hr />

## 数据集概况(更新中..)
### 一、ModelNet40(点云分类)
普林斯顿ModelNet项目的目标是为计算机视觉、计算机图形学、机器人和认知科学领域的研究者们提供一个全面、干净的三维CAD模型集合, 该数据的主页地址[https://modelnet.cs.princeton.edu](https://modelnet.cs.princeton.edu/), 数据最早发布在论文[3D ShapeNets: A Deep Representation for Volumetric Shapes](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf) [CVPR 2015]上.

相关工作人员从数据中选择了常见的40类和10类构成数组子集, 分别表示为ModelNet40和ModelNet10, 且两个数据集都有orientation  aligned的版本。实验中数据用到比较多的是ModelNet40, 有如下三种数据形式:

| 数据集 | modelnet40_normal_resampled.zip | modelnet40_ply_hdf5_2048.zip | ModelNet40.zip |
| :---: | :---: | :---: | :---: |
| 文件大小 | 1.71G | 435M | 2.04G |
| 内容 | point: x, y, z, normal_x, normal_y, normal_z;  <br /> shape: 10k points | point: x, y, z;  <br /> shape: 2048 points | off格式, 具体参考[这里](https://segeval.cs.princeton.edu/public/off_format.html) | 
| 训练集 / 测试集 | 9843 / 2468 | 9840 / 2468 | 9844 / 2468 |
| 下载地址 | [modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) | [modelnet40_ply_hdf5_2048.zip](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) | [ModelNet40.zip](http://modelnet.cs.princeton.edu/ModelNet40.zip) |

### 二、ShapeNet Part(点云分割)
ShapeNet数据集是一个有丰富标注的、大规模的3D图像数据集, 发布于[ShapeNet: An Information-Rich 3D Model Repository](https://arxiv.org/pdf/1512.03012.pdf) [arXiv 2015], 它是普林斯顿大学、斯坦福大学和TTIC研究人员共同努力的结果, 官方主页为[shapenet.org](https://www.shapenet.org/).ShapeNet包括ShapeNetCore和ShapeNetSem子数据集.

ShapeNet Part是从ShapeNetCore数据集选择了16类并进行语义信息标注的数据集, 用于点云的语义分割任务, 其数据集发表于[A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://www-cs.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf) [SIGGRAPH Asia 2016], 官方主页为 [ShapeNet Part](https://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html). 数据包含几个不同的版本, 其下载链接分别为[shapenetcore_partanno_v0.zip](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip) (1.08G)和[shapenetcore_partanno_segmentation_benchmark_v0.zip](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip)(635M). 下面就第2个数据集segmentation benchmark进行介绍:

从下面表格可以看出, ShapeNet Part总共有16类, 50个parts，总共包括16846个样本。该数据集中样本呈现出不均衡特性，比如Table包括5263个, 而Earphone只有69个。每个样本包含2000多个点, 属于小数据集。该数据集中**训练集**12137个, **验证集**1870个, **测试集**2874个, **总计**16881个。[注意, 这里和下面表格统计的(16846)并不一样, 后来发现是训练集、验证集和测试集有35个重复的样本]

| 类别 | nparts/shape | nsamples | 平均npoints/shape |
| :---: | :---: | :---: | :---: |
| Airplane | 4 | 2690 | 2577 |
| Bag | 2 | 76 | 2749 |
| Cap | 2 | 55 | 2631 |
| Car | 4 | 898 | 2763 |
| Chair | 4 | 3746 | 2705 |
| Earphone | 3 | 69 | 2496 |
| Guitar | 3 | 787 | 2353 |
| Knife | 2 | 392 | 2156 |
| Lamp | 4 | 1546 | 2198 |
| Laptop | 2 | 445 | 2757 |
| Motorbike | 6 | 202 | 2735 |
| Mug | 2 | 184 | 2816 |
| Pistol | 3 | 275 | 2654 |
| Rocket | 3 | 66 | 2358 |
| Skateboard | 3 | 152 | 2529 |
| Table | 3 | 5263 | 2722 |
| **Total** | **50** | **16846** | **2616** |


### 三、S3DIS(语义分割)
S3DIS是3D室内场景的数据集, 主要用于点云的语义分割任务。主页[http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html). (但官方主页我暂时访问不了了, 关于数据集背景的介绍性说明就不写了). 关于S3DIS的论文是[Joint 2D-3D-Semantic Data for Indoor Scene Understanding](https://arxiv.org/pdf/1702.01105.pdf) [arXiv 2017]和[3D Semantic Parsing of Large-Scale Indoor Spaces](http://svl.stanford.edu/assets/papers/3D_Semantic_Parsing.pdf) [CVPR 2016]. S3DIS从3个building的6个Area采集得到, Area1, Area3, Area6属于buidling 1, Area2和Area4属于building 2, Area5属于building 3. 常用的数据下载格式包括如下三种:
- [Stanford3dDataset_v1.2_Aligned_Version.zip](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1), 比如: [RandLA-Net](https://github.com/QingyongHu/RandLA-Net)
- [Stanford3dDataset_v1.2.zip](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1), 比如: [CloserLook3D](https://github.com/zeliu98/CloserLook3D/tree/master/pytorch)
- [indoor3d_sem_seg_hdf5_data.zip](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip), 比如: [PointNet](https://github.com/charlesq34/pointnet)

其中[Stanford3dDataset_v1.2_Aligned_Version.zip](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)和[Stanford3dDataset_v1.2.zip](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)都是完整场景的数据集, 每个点对应6个维度(x, y, z, r, g, b), 而[indoor3d_sem_seg_hdf5_data.zip](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip)是对原始数据场景的切割，把大场景切割成1m x 1m的block: 完整数据集被切割成了23585个block, 每个block是4096个点, 每个点对应9个维度: 除了x, y, z, r, g, b信息外，剩余的3维是相对于所在大场景的位置(归一化坐标).

下面是由[Stanford3dDataset_v1.2.zip](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)数据统计得到的关于S3DIS的信息, 可能和论文中一些结果不太一致。S3DIS数据集由以上6个Area采集得到, 共包含272个场景, 可分为11种不同的场景(括号内为场景数量, 场景大小(点的数量)): office(156, 87w), conference room(11, 142w), hallway(61, 122w), auditorium(2, 817w), open
space(1, 197w), lobby(3, 242w), lounge(3, 146w), pantry(3, 58w), copy room(2, 52w), storage(19, 35w) and WC(11, 70w). 根据语义信息, 上述场景被分成14个类别, 如下表所示. 可以看到不同的类别也是不均衡的, 比如wall有1547个, 但sofa只有55个.
| **Total** | column | clutter | chair | window | beam | floor | wall | ceiling | door | bookcase | board | table | sofa | stairs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9833** | 254 | 3882 | 1363 | 168 | 159 | 284 | 1547 | 385 | 543 | 584 | 137 | 455 | 55 | 17 |