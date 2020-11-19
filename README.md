# 3D - Point Cloud

**Paper list** and **Datasets** about Point Cloud. Datasets can be found in [Datasets.md](https://github.com/zhulf0804/3D-PointCloud/blob/master/Datasets.md).

<hr />

## Survey papers

- [Deep Learning for 3D Point Clouds: A Survey](https://arxiv.org/pdf/1912.12033.pdf) [TPAMI 2020]
- [A Comprehensive Performance Evaluation of 3D Local Feature Descriptors](https://www.researchgate.net/profile/Yulan_Guo/publication/274387466_A_Comprehensive_Performance_Evaluation_of_3D_Local_Feature_Descriptors/links/552b4a5b0cf29b22c9c1a6d2/A-Comprehensive-Performance-Evaluation-of-3D-Local-Feature-Descriptors.pdf) [IJCV 2016]

## 2020
- ECCV
	- [SoftPoolNet: Shape Descriptor for Point Cloud Completion and Classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480069.pdf) [`completion`, `cls`; [Github](https://github.com/wangyida/softpool)]
	- [Detail Preserved Point Cloud Completion via Separated Feature Aggregation](https://arxiv.org/pdf/2007.02374.pdf) [`completion`; [Tensorflow](https://github.com/XLechter/Detail-Preserved-Point-Cloud-Completion-via-SFA)]
	- [PointPWC-Net: A Coarse-to-Fine Network for Supervised and Self-Supervised Scene Flow Estimation on 3D Point Clouds](https://arxiv.org/pdf/1911.12408.pdf) [`flow estimation`; [PyTorch](https://github.com/DylanWusee/PointPWC)]
	- [JSENet: Joint Semantic Segmentation and Edge Detection Network for 3D Point Clouds](https://arxiv.org/pdf/2007.06888.pdf) [`seg`; [Tensorflow](https://github.com/hzykent/JSENet)]
	- [A Closer Look at Local Aggregation Operators in Point Cloud Analysis](https://arxiv.org/pdf/2007.01294.pdf) [`cls`, `seg`; [Code](https://github.com/zeliu98/CloserLook3D)]
	- [Instance-Aware Embedding for Point Cloud Instance Segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750256.pdf) [`seg`]
	- [Multimodal Shape Completion via Conditional Generative Adversarial Networks](https://arxiv.org/pdf/2003.07717.pdf) [`completion`; [PyTorch](https://github.com/ChrisWu1997/Multimodal-Shape-Completion)]
	- [GRNet: Gridding Residual Network for Dense Point Cloud Completion](https://arxiv.org/pdf/2006.03761.pdf) [`completion`; [PyTorch](https://github.com/hzxie/GRNet)]
	- [3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection](https://arxiv.org/pdf/2004.12636.pdf) [`det`]
	- [SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds](https://arxiv.org/pdf/2004.02774.pdf) [`det`; [Github](https://github.com/xinge008/SSN)]
	- [Pillar-based Object Detection for Autonomous Driving](https://arxiv.org/pdf/2007.10323.pdf) [`det`, `autonomous driving`; [Tensorflow](https://github.com/WangYueFt/pillar-od)]
	- [EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection](https://arxiv.org/pdf/2007.08856.pdf) [`det`; [PyTorch](https://github.com/happinesslz/EPNet)]
	- [Finding Your (3D) Center: 3D Object Detection Using a Learned Loss](https://arxiv.org/pdf/2004.02693.pdf) [`det`; [Tensorflow](https://github.com/dgriffiths3/finding-your-center)]
	- [Weakly Supervised 3D Object Detection from Lidar Point Cloud](https://arxiv.org/pdf/2007.11901.pdf) [`det`; [PyTorch](https://github.com/hlesmqh/WS3D)]
	- [H3DNet: 3D Object Detection Using Hybrid Geometric Primitives](https://arxiv.org/pdf/2006.05682.pdf) [`det`; [Tensorflow](https://github.com/zaiweizhang/H3DNet)]
	- [Generative Sparse Detection Networks for 3D Single-shot Object Detection](https://arxiv.org/pdf/2006.12356.pdf) [`det`; [Github](https://github.com/jgwak/GSDN)]
	- [Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution](https://arxiv.org/pdf/2007.16100.pdf) [`seg`, `det`; [PyTorch](https://github.com/mit-han-lab/e3d)]
	- [DeepGMR: Learning Latent Gaussian Mixture Models for Registration](https://arxiv.org/pdf/2008.09088.pdf) [`registration`; [PyTorch](https://github.com/wentaoyuan/deepgmr)]
	- [Quaternion Equivariant Capsule Networks for 3D Point Clouds](https://arxiv.org/pdf/1912.12098.pdf) [[PyTorch](https://github.com/tolgabirdal/qecnetworks)]
	- [PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding](https://arxiv.org/pdf/2007.10985.pdf) [`unsupervised`; `cls`, `seg`, `det`; [PyTorch](https://github.com/facebookresearch/PointContrast)]
	- [Convolutional Occupancy Networks](https://arxiv.org/pdf/2003.04618.pdf) [`reconstruction`; [PyTorch](https://github.com/autonomousvision/convolutional_occupancy_networks)]
	- [Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration](https://arxiv.org/pdf/1910.10328.pdf) [`registration`; [PyTorch](https://github.com/jiahaowork/idam)]
	- [Progressive Point Cloud Deconvolution Generation Network](https://arxiv.org/pdf/2007.05361.pdf) [`generation`; [github](https://github.com/fpthink/PDGN)]
	- [Reinforced Axial Refinement Network for Monocular 3D Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620528.pdf) [`det`, `monocular`]
	- [Monocular 3D Object Detection via Feature Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540018.pdf) [`det`, `monocular`]
	- [Improving 3D Object Detection through Progressive Population Based Augmentation](https://arxiv.org/pdf/2004.00831.pdf) [`det`]
	- [An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds](https://arxiv.org/pdf/2007.12392.pdf) [`det`]
	- [Rotation-robust Intersection over Union for 3D Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650460.pdf)
- CVPR
	- [Deep Global Registration](https://arxiv.org/pdf/2004.11540.pdf) [`registration`; [PyTorch](https://github.com/chrischoy/DeepGlobalRegistration)]
	- [Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.pdf) [`det`; [Tensorflow](https://github.com/WeijingShi/Point-GNN)]
	- [ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_ImVoteNet_Boosting_3D_Object_Detection_in_Point_Clouds_With_Image_CVPR_2020_paper.pdf) [`det`]
	- [OccuSeg: Occupancy-aware 3D Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_OccuSeg_Occupancy-Aware_3D_Instance_Segmentation_CVPR_2020_paper.pdf) [`seg`]
	- [Fusion-Aware Point Convolution for Online Semantic 3D Scene Segmentation](https://arxiv.org/pdf/2003.06233.pdf) [`seg`; [PyTorch](https://github.com/jzhzhang/FusionAwareConv)]
	- [MLCVNet: Multi-Level Context VoteNet for 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_MLCVNet_Multi-Level_Context_VoteNet_for_3D_Object_Detection_CVPR_2020_paper.pdf) [`det`; [PyTorch](https://github.com/NUAAXQ/MLCVNet)]
	- [Going Deeper with Lean Point Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Going_Deeper_With_Lean_Point_Networks_CVPR_2020_paper.pdf) [`seg`; [PyTorch](https://github.com/erictuanle/GoingDeeperwPointNetworks)]
	- [Point Cloud Completion by Skip-attention Network with Hierarchical Folding](https://arxiv.org/pdf/2005.03871.pdf) [`completion`]
	- [Unsupervised Learning of Intrinsic Structural Representation Points](https://arxiv.org/pdf/2003.01661.pdf) [[PyTorch](https://github.com/NolenChen/3DStructurePoints)]
	- [PF-Net: Point Fractal Network for 3D Point Cloud Completion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf) [`completion`; [PyTorch](https://github.com/zztianzz/PF-Net-Point-Fractal-Network)]
	- [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.pdf) [`det`; [code](https://github.com/open-mmlab/OpenPCDet)]
	- [Adaptive Hierarchical Down-Sampling for Point Cloud Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Nezhadarya_Adaptive_Hierarchical_Down-Sampling_for_Point_Cloud_Classification_CVPR_2020_paper.pdf) [`downsampling`, `cls`]
	- [SA-SSD: Structure Aware Single-stage 3D Object Detection from Point Cloud](https://www4.comp.polyu.edu.hk/~cslzhang/paper/SA-SSD.pdf) [`det`; [PtTorch](https://github.com/skyhehe123/SA-SSD)]
	- [3DRegNet: A Deep Neural Network for 3D Point Registration](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pais_3DRegNet_A_Deep_Neural_Network_for_3D_Point_Registration_CVPR_2020_paper.pdf) [`registration`; [Tensorflow](https://github.com/3DVisionISR/3DRegNet)]
	- [MINA: Convex Mixed-Integer Programming for Non-Rigid Shape Alignment](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bernard_MINA_Convex_Mixed-Integer_Programming_for_Non-Rigid_Shape_Alignment_CVPR_2020_paper.pdf) [`non-rigid alignment`]
	- [SampleNet: Differentiable Point Cloud Sampling](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.pdf) [`sample`, `cls`, `registration`, `reconstruction`; [PyTorch](https://github.com/itailang/SampleNet)]
	- [Learning multiview 3D point cloud registration](https://arxiv.org/pdf/2001.05119.pdf) [`multiview registration`; [PyTorch](https://github.com/zgojcic/3D_multiview_reg)]
	- [Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences](https://arxiv.org/pdf/2005.01014.pdf) [`registration`; [PyTorch](https://github.com/XiaoshuiHuang/fmr)] 
 	- [PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling](https://arxiv.org/pdf/2003.00492.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/yanx27/PointASNL)]
	- [Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds](https://arxiv.org/pdf/2003.12971.pdf) [`unsupervised`; `cls`; [PyTorch](https://github.com/raoyongming/PointGLR)]
	- [Grid-GCN for Fast and Scalable Point Cloud Learning](https://arxiv.org/pdf/1912.02984.pdf) [`cls`, `seg`; [mxnet](https://github.com/Xharlie/Grid-GCN)]
	- [FPConv: Learning Local Flattening for Point Convolution](https://arxiv.org/pdf/2002.10701.pdf) [`cls`, `seg`; [PyTorch](https://github.com/lyqun/FPConv)]
	- [PointAugment: an Auto-Augmentation Framework for Point Cloud Classification](https://arxiv.org/pdf/2002.10876.pdf) [`cls`, `retrieval`; [github](https://github.com/liruihui/PointAugment/)]
	- [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/pdf/1911.11236.pdf) [`seg`; [Tensorflow](https://github.com/QingyongHu/RandLA-Net)]
	- [Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels](https://arxiv.org/pdf/2004.04091.pdf) [`weakly supervised`; `seg`; [Tensorflow](https://github.com/alex-xun-xu/WeakSupPointCloudSeg)]
	- [PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation](https://arxiv.org/pdf/2003.14032.pdf) [`seg`; [PyTorch](https://github.com/edwardzhou130/PolarSeg)]
	- [Learning to Segment 3D Point Clouds in 2D Image Space](https://arxiv.org/pdf/2003.05593.pdf) [`seg`; [Keras](https://github.com/Zhang-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space)]
	- [PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation](https://arxiv.org/pdf/2004.01658.pdf) [`seg`; [PyTorch](https://github.com/Jia-Research-Lab/PointGroup)]
	- [D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features](https://arxiv.org/pdf/2003.03164.pdf) [`keypoints`, `registration`; [Tensorflow](https://github.com/XuyangBai/D3Feat), [PyTorch](https://github.com/XuyangBai/D3Feat.pytorch)]
	- [RPM-Net: Robust Point Matching using Learned Features](https://arxiv.org/pdf/2003.13479.pdf) [`registration`; [PyTorch](https://github.com/yewzijian/RPMNet)]
	- [Cascaded Refinement Network for Point Cloud Completion](https://arxiv.org/pdf/2004.03327.pdf) [`completion`; [Tensorflow](https://github.com/xiaogangw/cascaded-point-completion)]
	- [P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds](https://arxiv.org/pdf/2005.13888.pdf) [`tracking`; [PyTorch](https://github.com/HaozheQi/P2B)]
	- [An Efficient PointLSTM for Point Clouds Based Gesture Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Min_An_Efficient_PointLSTM_for_Point_Clouds_Based_Gesture_Recognition_CVPR_2020_paper.pdf) [`gesture`; [PyTorch](https://github.com/Blueprintf/pointlstm-gesture-recognition-pytorch)]

- Others
	- [Self-Supervised Few-Shot Learning on Point Clouds](https://arxiv.org/pdf/2009.14168.pdf) [`cls`, `seg`; NeurIPS]
	- [Rotation-Invariant Local-to-Global Representation Learning for 3D Point Cloud](https://arxiv.org/pdf/2010.03318.pdf) [`cls`; NeurIPS]
	- [PIE-NET: Parametric Inference of Point Cloud Edges](https://arxiv.org/pdf/2007.04883.pdf) [`edge det`; NeurIPS]
	- [Unpaired Point Cloud Completion on Real Scans using Adversarial Training](https://arxiv.org/pdf/1904.00069.pdf) [`completion`; [Tensorflow](https://github.com/xuelin-chen/pcl2pcl-gan-pub); ICLR]
	- [AdvectiveNet: An Eulerian-Lagrangian Fluidic Reservoir for Point Cloud Processing](https://arxiv.org/pdf/2002.00118.pdf) [`cls`, `seg`; [PyTorch](https://github.com/xingzhehe/AdvectiveNet-An-Eulerian-Lagrangian-Fluidic-Reservoir-for-Point-Cloud-Processing); ICLR]
	- [Tranquil Clouds: Neural Networks for Learning Temporally Coherent Features in Point Clouds](https://arxiv.org/pdf/1907.05279.pdf) [ICLR]
	- [MSN: Morphing and Sampling Network for Dense Point Cloud Completion](https://cseweb.ucsd.edu/~mil070/projects/AAAI2020/paper.pdf) [`completion`; [PyTorch](https://github.com/Colin97/MSN-Point-Cloud-Completion); AAAI]
	- [TANet: Robust 3D Object Detection from Point Clouds with Triple Attention](https://arxiv.org/pdf/1912.05163.pdf) [`det`; [PyTorch](https://github.com/happinesslz/TANet); AAAI]
	- [JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds](https://arxiv.org/pdf/1912.09654.pdf) [`seg`; [Tensorflow](https://github.com/dlinzhao/JSNet)]
	- [Point2Node: Correlation Learning of Dynamic-Node for Point Cloud Feature Modeling](https://arxiv.org/pdf/1912.10775.pdf) [`cls`, `seg`; AAAI]
	- [Pointwise Rotation-Invariant Network with Adaptive Sampling and 3D Spherical Voxel Convolution](https://arxiv.org/pdf/1811.09361.pdf) [`cls`, `seg`, `matching`; AAAI]
	- [Differentiable Manifold Reconstruction for Point Cloud Denoising](https://arxiv.org/pdf/2007.13551.pdf) [`denoising`; [PyTorch](https://github.com/luost26/DMRDenoise); ACM MM]
	- [Weakly Supervised 3D Object Detection from Point Clouds](https://arxiv.org/pdf/2007.13970.pdf) [`det`; [Tensorflow](https://github.com/Zengyi-Qin/Weakly-Supervised-3D-Object-Detection); ACM MM]
	- [Unsupervised Detection of Distinctive Regions on 3D Shapes](https://arxiv.org/pdf/1905.01684.pdf) [`unsupervised`; [Tensorflow](https://github.com/nini-lxz/Unsupervised-Shape-Distinction-Detection); TOG]
	- [Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds](https://arxiv.org/pdf/1907.12046.pdf) [`seg`, `cls`; [Project](https://francisengelmann.github.io/DPC/); ICRA]
	- [Fast and Automatic Registration of Terrestrial Point Clouds Using 2D Line Features](https://www.researchgate.net/publication/340771096_Fast_and_Automatic_Registration_of_Terrestrial_Point_Clouds_Using_2D_Line_Features/fulltext/5ea05e5545851564fc34cb0c/Fast-and-Automatic-Registration-of-Terrestrial-Point-Clouds-Using-2D-Line-Features.pdf) [`registration`; Remote Sensing]
	- [ConvPoint: Continuous Convolutions for Point Cloud Processing](https://arxiv.org/pdf/1904.02375.pdf) [`cls`, `seg`; [PyTorch](https://github.com/aboulch/ConvPoint); Computers & Graphics]
- arXiv
	- [Learning 3D-3D Correspondences for One-shot Partial-to-partial Registration](https://arxiv.org/pdf/2006.04523.pdf) [`registration`]
	- [PRE-TRAINING BY COMPLETING POINT CLOUDS](https://arxiv.org/pdf/2010.01089.pdf) [`pre-training`, `cls`, `seg`; [Github](https://github.com/hansen7/OcCo)]
	- [Continuous Geodesic Convolutions for Learning on 3D Shapes](https://arxiv.org/pdf/2002.02506.pdf) [`descriptor`, `match`, `seg`]
	- [Multi-Resolution Graph Neural Network for Large-Scale Pointcloud Segmentation](https://arxiv.org/pdf/2009.08924.pdf) [`seg`]
	- [A Density-Aware PointRCNN for 3D Objection Detection in Point Clouds](https://arxiv.org/pdf/2009.05307.pdf) [`det`]
	- [TEASER: Fast and Certifiable Point Cloud Registration](https://arxiv.org/pdf/2001.07715.pdf) [`registration`; [Github](https://github.com/MIT-SPARK/TEASER-plusplus)]
	- [Part-Aware Data Augmentation for 3D Object Detection in Point Cloud](https://arxiv.org/pdf/2007.13373.pdf) [`det`, `augmentation`; [PyTorch](https://github.com/sky77764/pa-aug.pytorch)]

## 2019
- ICCV
	- [USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.pdf) [`keypoints`, `registration`; [PyTorch](https://github.com/lijx10/USIP)]
	- [Unsupervised Multi-Task Feature Learning on Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hassani_Unsupervised_Multi-Task_Feature_Learning_on_Point_Clouds_ICCV_2019_paper.pdf) [`cls`, `seg`]
	- [Multi-Angle Point Cloud-VAE: Unsupervised Feature Learning for 3D Point Clouds from Multiple Angles by Joint Self-Reconstruction and Half-to-Half Prediction](https://arxiv.org/pdf/1907.12704.pdf) [`unsupervised`, `cls`, `generation`, `seg`, `completion`]
	- [SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences](https://openaccess.thecvf.com/content_ICCV_2019/papers/Behley_SemanticKITTI_A_Dataset_for_Semantic_Scene_Understanding_of_LiDAR_Sequences_ICCV_2019_paper.pdf) [`dataset`]
	- [MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_MeteorNet_Deep_Learning_on_Dynamic_3D_Point_Cloud_Sequences_ICCV_2019_paper.pdf) [`cls`, `seg`, `flow estimation`; [Tensorflow](https://github.com/xingyul/meteornet)]
	- [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_DeepGCNs_Can_GCNs_Go_As_Deep_As_CNNs_ICCV_2019_paper.pdf) [`seg`; [Tensorflow](https://github.com/lightaime/deep_gcns)]
	- [VV-NET: Voxel VAE Net with Group Convolutions for Point Cloud Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Meng_VV-Net_Voxel_VAE_Net_With_Group_Convolutions_for_Point_Cloud_ICCV_2019_paper.pdf) [`seg`; [Github](https://github.com/xianyuMeng/VV-Net-Voxel-VAE-Net-with-Group-Convolutions-for-Point-Cloud-Segmentation)]
	- [Interpolated Convolutional Networks for 3D Point Cloud Understanding](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mao_Interpolated_Convolutional_Networks_for_3D_Point_Cloud_Understanding_ICCV_2019_paper.pdf) [`cls`, `seg`]
	- [Dynamic Points Agglomeration for Hierarchical Point Sets Learning](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Dynamic_Points_Agglomeration_for_Hierarchical_Point_Sets_Learning_ICCV_2019_paper.pdf) [`cls`, `seg`]
	- [ShellNet: Efficient Point Cloud Convolutional Neural Networks using Concentric Shells Statistics](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_ShellNet_Efficient_Point_Cloud_Convolutional_Neural_Networks_Using_Concentric_Shells_ICCV_2019_paper.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/hkust-vgd/shellnet)]
	- [Fast Point R-CNN](https://arxiv.org/pdf/1908.02990.pdf) [`det`]
	- [Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://openaccess.thecvf.com/content_ICCV_2019/papers/Uy_Revisiting_Point_Cloud_Classification_A_New_Benchmark_Dataset_and_Classification_ICCV_2019_paper.pdf) [`dataset`; `cls`; [Tensorflow](https://github.com/hkust-vgd/scanobjectnn)]
	- [KPConv: Flexible and Deformable Convolution for Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf) [`cls`, `seg`; [code](https://github.com/HuguesTHOMAS/KPConv)]
	- [Fully Convolutional Geometric Features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choy_Fully_Convolutional_Geometric_Features_ICCV_2019_paper.pdf) [`match`; [PyTorch](https://github.com/chrischoy/FCGF)]
	- [Deep Closest Point: Learning Representations for Point Cloud Registration](https://arxiv.org/pdf/1905.03304.pdf) [`registration`; [PyTorch](https://github.com/WangYueFt/dcp)]
	- [DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration](https://arxiv.org/pdf/1905.04153.pdf) [`registration`]
	- [Efficient and Robust Registration on the 3D Special Euclidean Group](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bhattacharya_Efficient_and_Robust_Registration_on_the_3D_Special_Euclidean_Group_ICCV_2019_paper.pdf) [`registration`]
	- [Hierarchical Point-Edge Interaction Network for Point Cloud Semantic Segmentation](https://arxiv.org/pdf/1909.10469.pdf) [`seg`]
	- [DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing](https://arxiv.org/pdf/1909.03669.pdf) [`cls`, `retrieval`, `seg`, `normal estimation`; [PyTorch](https://github.com/Yochengliu/DensePoint)]
	- [DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense](https://arxiv.org/pdf/1812.11017.pdf) [`cls`]
	- [Efficient Learning on Point Clouds with Basis Point Sets](https://arxiv.org/pdf/1908.09186.pdf) [`cls`, `registration`; [PyTorch](https://github.com/sergeyprokudin/bps)]
	- [PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows](https://arxiv.org/pdf/1906.12320.pdf) [`generation`, `reconstruction`; [Pytorch](https://github.com/stevenygd/PointFlow)
	- [PU-GAN: a Point Cloud Upsampling Adversarial Network](https://arxiv.org/pdf/1907.10844) [`upsampling`, `reconstruction`; [Project](https://liruihui.github.io/publication/PU-GAN/)]
	- [3D Point Cloud Learning for Large-scale Environment Analysis and Place Recognition](https://arxiv.org/pdf/1812.07050.pdf) [`retrieval`, `place recognition`]
	- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/pdf/1904.09664.pdf) [`det`; [PyTorch](https://github.com/facebookresearch/votenet)]
	- [Exploring the Limitations of Behavior Cloning for Autonomous Driving](https://arxiv.org/pdf/1904.08980.pdf) [`autonomous driving`; [Pytorch](https://github.com/felipecode/coiltraine)]

- CVPR
	- [Associatively Segmenting Instances and Semantics in Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Associatively_Segmenting_Instances_and_Semantics_in_Point_Clouds_CVPR_2019_paper.pdf) [`seg`; [Tensorflow](https://github.com/WXinlong/ASIS)]
	- [3D Point Capsule Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_3D_Point_Capsule_Networks_CVPR_2019_paper.pdf) [`autoencoder`; [PyTorch](https://github.com/yongheng1991/3D-point-capsule-networks)]
	- [Patch-based Progressive 3D Point Set Upsampling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yifan_Patch-Based_Progressive_3D_Point_Set_Upsampling_CVPR_2019_paper.pdf) [`upsampling`; [Tensorflow](https://github.com/yifita/3PU), [PyTorch](https://github.com/yifita/3PU_pytorch)]
	- [Generating 3D Adversarial Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiang_Generating_3D_Adversarial_Point_Clouds_CVPR_2019_paper.pdf) [`adversary`; [Tensorflow](https://github.com/xiangchong1/3d-adv-pc)]
	- [RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sarmad_RL-GAN-Net_A_Reinforcement_Learning_Agent_Controlled_GAN_Network_for_Real-Time_CVPR_2019_paper.pdf) [`completion`; [PyTorch](https://github.com/iSarmad/RL-GAN-Net)]
	- [GSPN: Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_GSPN_Generative_Shape_Proposal_Network_for_3D_Instance_Segmentation_in_CVPR_2019_paper.pdf) [`seg`; [Tensorflow](https://github.com/ericyi/GSPN)]
	- [JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pham_JSIS3D_Joint_Semantic-Instance_Segmentation_of_3D_Point_Clouds_With_Multi-Task_CVPR_2019_paper.pdf) [`seg`; [PyTorch](https://github.com/pqhieu/jsis3d)]
	- [3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_3D-SIS_3D_Semantic_Instance_Segmentation_of_RGB-D_Scans_CVPR_2019_paper.pdf) [`seg`; [PyTorch](https://github.com/Sekunde/3D-SIS)]
	- [Learning Transformation Synchronization](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Learning_Transformation_Synchronization_CVPR_2019_paper.pdf) [`transformation  synchronization`, `registration`; [PyTorch](https://github.com/xiangruhuang/Learning2Sync)]
	- [SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences](https://openaccess.thecvf.com/content_CVPR_2019/papers/Le_SDRSAC_Semidefinite-Based_Randomized_Approach_for_Robust_Point_Cloud_Registration_Without_CVPR_2019_paper.pdf) [`registration`; [Github](https://github.com/intellhave/SDRSAC)]
	- [Learning Transformation Synchronization](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Learning_Transformation_Synchronization_CVPR_2019_paper.pdf) [`reconstruction`; [PyTorch](https://github.com/xiangruhuang/Learning2Sync)]
	- [3D Local Features for Direct Pairwise Registration](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf) [`registration`]
	- [DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_DeepMapping_Unsupervised_Map_Estimation_From_Multiple_Point_Clouds_CVPR_2019_paper.pdf) [`registration`; [Github](https://github.com/ai4ce/DeepMapping)]
	- [Relation-Shape Convolutional Neural Network for Point Cloud Analysis](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Relation-Shape_Convolutional_Neural_Network_for_Point_Cloud_Analysis_CVPR_2019_paper.pdf) [`cls`, `seg`, `normal estimation`; [PyTorch](https://github.com/Yochengliu/Relation-Shape-CNN)]
	- [Modeling Local Geometric Structure of
	3D Point Clouds using Geo-CNN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lan_Modeling_Local_Geometric_Structure_of_3D_Point_Clouds_Using_Geo-CNN_CVPR_2019_paper.pdf) [`cls`, `det`; [Tensorflow](https://github.com/voidrank/Geo-CNN)]
	- [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) [`seg`; [PyTorch](https://github.com/chrischoy/SpatioTemporalSegmentation)]
	- [PCAN: 3D Attention Map Learning Using Contextual Information for Point Cloud Based Retrieval](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_PCAN_3D_Attention_Map_Learning_Using_Contextual_Information_for_Point_CVPR_2019_paper.pdf) [`retrieval`; [Tensorflow](https://github.com/XLechter/PCAN)]
	- [Attentional PointNet for 3D-Object Detection in Point Clouds](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Paigwar_Attentional_PointNet_for_3D-Object_Detection_in_Point_Clouds_CVPRW_2019_paper.pdf) [`det`; [PyTorch](https://github.com/anshulpaigwar/Attentional-PointNet)]
	- [Octree guided CNN with Spherical Kernels for 3D Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lei_Octree_Guided_CNN_With_Spherical_Kernels_for_3D_Point_Clouds_CVPR_2019_paper.pdf) [`cls`, `seg`; [Github](https://github.com/hlei-ziyan/psicnn)]
	- [A-CNN: Annularly Convolutional Neural Networks on Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Komarichev_A-CNN_Annularly_Convolutional_Neural_Networks_on_Point_Clouds_CVPR_2019_paper.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/artemkomarichev/a-cnn)]
	- [ClusterNet: Deep Hierarchical Cluster Network with Rigorously Rotation-Invariant Representation for Point Cloud Analysis](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_ClusterNet_Deep_Hierarchical_Cluster_Network_With_Rigorously_Rotation-Invariant_Representation_for_CVPR_2019_paper.pdf) [`cls`]
	- [Graph Attention Convolution for Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Graph_Attention_Convolution_for_Point_Cloud_Semantic_Segmentation_CVPR_2019_paper.pdf) [`seg`; [PyTorch-unofficial](https://github.com/yanx27/GACNet)]
	- [PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_PointWeb_Enhancing_Local_Neighborhood_Features_for_Point_Cloud_Processing_CVPR_2019_paper.pdf) [`seg`, `cls`; [PyTorch](https://github.com/hszhao/PointWeb)]
	- [Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Modeling_Point_Clouds_With_Self-Attention_and_Gumbel_Subset_Sampling_CVPR_2019_paper.pdf) [`cls`, `seg`, `gesture`]
	- [Learning to Sample](https://arxiv.org/pdf/1812.01659.pdf) [`sample`, `cls`, `retrieval`, `reconstruction`; [Tensorflow](https://github.com/orendv/learning_to_sample)]
	- [PointConv: Deep Convolutional Networks on 3D Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/DylanWusee/pointconv)]
	- [The Perfect Match: 3D Point Cloud Matching With Smoothed Densities](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gojcic_The_Perfect_Match_3D_Point_Cloud_Matching_With_Smoothed_Densities_CVPR_2019_paper.pdf) [`match`; [code](https://github.com/zgojcic/3DSmoothNet)]
	- [PointNetLK: Point Cloud Registration using PointNet](https://arxiv.org/pdf/1903.05711.pdf) [`registration`; [PyTorch](https://github.com/hmgoforth/PointNetLK)]
	- [PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf) [`det`; [PyTorch](https://github.com/sshaoshuai/PointRCNN)]		
	- [PointPillars: Fast Encoders for Object Detection From Point Clouds](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf) [`det`; [Pytorch](https://github.com/nutonomy/second.pytorch)]
	- [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/1812.07179.pdf) [`depth estimation`, `det`; [github](https://github.com/mileyan/pseudo_lidar)]
	- [ApolloCar3D: A Large 3D Car Instance Understanding Benchmark for Autonomous Driving](https://arxiv.org/pdf/1811.12222.pdf) [`dataset`, `autonomous driving`]
	- [Stereo R-CNN based 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/1902.09738.pdf) [`det`, `autonomous driving`; [github](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN)]
	- [Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction](https://arxiv.org/pdf/1904.01690.pdf) [`det`, `autonomous driving`; [Tesorflow](https://github.com/kujason/monopsr)]
	- [LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://arxiv.org/pdf/1903.08701.pdf) [`det`]
	- [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](https://arxiv.org/pdf/1903.10955.pdf) [`det`, `autonomous driving`]
	- [L3-Net: Towards Learning based LiDAR Localization for Autonomous Driving](https://songshiyu01.github.io/pdf/L3Net_W.Lu_Y.Zhou_S.Song_CVPR2019.pdf) [`autonomous driving`]
	- [Iterative Transformer Network for 3D Point Cloud](https://arxiv.org/pdf/1811.11209.pdf) [`pose`, `cls`, `seg`; [Tensorflow](https://github.com/wentaoyuan/it-net)]

- Others
	- [Self-Supervised Deep Learning on Point Clouds by Reconstructing Space](https://papers.nips.cc/paper/2019/file/993edc98ca87f7e08494eec37fa836f7-Paper.pdf) [`self-supervised, cls, seg`; NeurIPS]
	- [Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://arxiv.org/pdf/1906.01140.pdf) [`seg`; [Tensorflow](https://github.com/Yang7879/3D-BoNet); NeurIPS]
	- [PRNet: Self-Supervised Learning for Partial-to-Partial Registration](http://papers.nips.cc/paper/9085-prnet-self-supervised-learning-for-partial-to-partial-registration.pdf) [`registration`, `cls`; [PyTorch](https://github.com/WangYueFt/prnet); NeurIPS]
	- [Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/pdf/1907.03739.pdf) [`seg`, `det`; [PyTorch](https://github.com/mit-han-lab/pvcnn); NeurIPS]
	- [A Polynomial-time Solution for Robust Registration with Extreme Outlier Rates](https://arxiv.org/pdf/1903.08588.pdf) [`registration`; RSS]
	- [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/pdf/1801.07829.pdf) [`cls`, `seg`; [Github](https://github.com/WangYueFt/dgcnn); TOG]
	- [SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud](https://arxiv.org/pdf/1809.08495.pdf) [`seg`; [Tensorflow](https://github.com/xuanyuzhou98/SqueezeSegV2); ICRA]
	- [RangeNet++: Fast and Accurate LiDAR Semantic Segmentation](https://www.researchgate.net/profile/Cyrill_Stachniss/publication/338937998_RangeNet_Fast_and_Accurate_LiDAR_Semantic_Segmentation/links/5f609d4ba6fdcc1164133ac9/RangeNet-Fast-and-Accurate-LiDAR-Semantic-Segmentation.pdf) [`seg`; [PyTorch](https://github.com/PRBonn/lidar-bonnetal); IROS]
	- [AlignNet-3D: Fast Point Cloud Registration of Partially Observed Objects](https://arxiv.org/pdf/1910.04668.pdf) [`registration`; [Tensorflow](https://github.com/grossjohannes/AlignNet-3D); 3DV]

- arXiv
	- [PCRNet: Point Cloud Registration Network using PointNet Encoding](https://arxiv.org/pdf/1908.07906.pdf) [`registration`; [PyTorch](https://github.com/vinits5/pcrnet_pytorch), [Tensorflow](https://github.com/vinits5/pcrnet)]
	- [LSANet: Feature Learning on Point Sets by Local Spatial Aware Layer](https://arxiv.org/pdf/1905.05442.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/LinZhuoChen/LSANet)]
	- [Adversarial Sensor Attack on LiDAR-based Perception in Autonomous Driving](https://arxiv.org/pdf/1907.06826.pdf) [`autonomous driving`]

## 2018
- CVPR
	- [Deep Parametric Continuous Convolutional Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf) [`seg`, `motion estimation(lidar flow)`]
	- [Attentional ShapeContextNet for Point Cloud Recognition](http://pages.ucsd.edu/~ztu/publication/cvpr18_ascn.pdf) [`cls`, `seg`]
	- [A Papier-Mâché Approach to Learning 3D Surface Generation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Groueix_A_Papier-Mache_Approach_CVPR_2018_paper.pdf) [`generation`; [PyTorch](https://github.com/ThibaultGROUEIX/AtlasNet)]
	- [Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf) [`seg`; [PyTorch](https://github.com/loicland/superpoint_graph)]
	- [FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_FoldingNet_Point_Cloud_CVPR_2018_paper.pdf) [`autoencoder`, `unsupervised`; [code](https://www.merl.com/research/license#FoldingNet)]
	- [FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis](https://arxiv.org/pdf/1706.05206.pdf) [`correspondence`, `seg`; [Tensorflow](https://github.com/nitika-verma/FeaStNet)]
	- [PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition](https://openaccess.thecvf.com/content_cvpr_2018/papers/Uy_PointNetVLAD_Deep_Point_CVPR_2018_paper.pdf) [`retrieval`, `place recognition`; [Tensorflow](https://github.com/mikacuy/pointnetvlad)]
	- [PU-Net: Point Cloud Upsampling Network](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.pdf) [`upsampling`; [Tensorflow](https://github.com/yulequan/PU-Net)]
	- [SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf) [`seg`; [Tensorflow](https://github.com/laughtervv/SGPN)]
	- [Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1573.pdf) [`cls`, `seg`; [Code](http://www.merl.com/research/license#KCNet)]
	- [Tangent Convolutions for Dense Prediction in 3D](http://vladlen.info/papers/tangent-convolutions.pdf) [`seg`; [Tensorflow](https://github.com/tatarchm/tangent_conv)]
	- [PointGrid: A Deep Network for 3D Shape Understanding](https://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/trucleduc/PointGrid)]
	- [3D Semantic Segmentation with Submanifold Sparse Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf) [`seg`; [Github](https://github.com/facebookresearch/SparseConvNet)]
	- [Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf) [`seg`; [PyTorch](https://github.com/loicland/superpoint_graph)]
	- [SPLATNet: Sparse Lattice Networks for Point Cloud Processing](https://openaccess.thecvf.com/content_cvpr_2018/papers/Su_SPLATNet_Sparse_Lattice_CVPR_2018_paper.pdf) [`seg`; [Caffe](https://github.com/NVlabs/splatnet)]
	- [Pointwise Convolutional Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hua_Pointwise_Convolutional_Neural_CVPR_2018_paper.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/hkust-vgd/pointwise)]
	- [SO-Net: Self-Organizing Network for Point Cloud Analysis](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf) [`autoencoder`, `cls`, `seg`; [PyTorch](https://github.com/lijx10/SO-Net)]
	- [Recurrent Slice Networks for 3D Segmentation of Point Clouds](https://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Recurrent_Slice_Networks_CVPR_2018_paper.pdf) [`seg`; [PyTorch](https://github.com/qianguih/RSNet/blob/master/README.md)]
	- [PPFNet: Global Context Aware Local Features for Robust 3D Point Matching](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_PPFNet_Global_Context_CVPR_2018_paper.pdf) [`registration`]
	- [PIXOR: Real-Time 3D Object Detection From Point Clouds](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf) [`det`; [PyTorch](https://github.com/philip-huang/PIXOR)]
	- [Frustum PointNets for 3D Object Detection From RGB-D Data](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf) [`det`; [Tensorflow](https://github.com/charlesq34/frustum-pointnets)]		
	- [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf) [`det`]
	- [3D-RCNN: Instance-Level 3D Object Reconstruction via Render-and-Compare](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kundu_3D-RCNN_Instance-Level_3D_CVPR_2018_paper.pdf) [`reconstruction`]
	- [Multi-Level Fusion Based 3D Object Detection From Monocular Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Multi-Level_Fusion_Based_CVPR_2018_paper.pdf) [`det`]

- ECCV

	- [SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Xu_SpiderCNN_Deep_Learning_ECCV_2018_paper.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/xyf513/SpiderCNN)]
	- [3DContextNet: K-d Tree Guided Hierarchical Learning of Point Clouds Using Local and Global Contextual Cues](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Zeng_3DContextNet_K-d_Tree_Guided_Hierarchical_Learning_of_Point_Clouds_Using_ECCVW_2018_paper.pdf) [`seg`, `cls`]
	- [Multiresolution Tree Networks for
	3D Point Cloud Processing](https://openaccess.thecvf.com/content_ECCV_2018/papers/Matheus_Gadelha_Multiresolution_Tree_Networks_ECCV_2018_paper.pdf) [`cls`, `generation`; [PyTorch](https://github.com/matheusgadelha/MRTNet)]
	- [HGMR: Hierarchical Gaussian Mixtures for
	Adaptive 3D Registration](https://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf) [`registration`; [unofficial code](https://github.com/somanshu25/GPU-Accelerated-Point-Cloud-Registration-Using-Hierarchical-GMM)]
	- [EC-Net: an Edge-aware Point set Consolidation Network](https://arxiv.org/pdf/1807.06010.pdf) [`consolidation`; [Tensorflow](https://github.com/yulequan/EC-Net)]
	- [Learning and Matching Multi-View Descriptors for Registration of Point Clouds](https://openaccess.thecvf.com/content_ECCV_2018/papers/Lei_Zhou_Learning_and_Matching_ECCV_2018_paper.pdf) [`registration`]
	- [Local Spectral Graph Convolution for Point Set Feature Learning](https://arxiv.org/pdf/1803.05827.pdf) [`cls`, `seg`]
	- [3D Recurrent Neural Networks with Context Fusion for Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoqing_Ye_3D_Recurrent_Neural_ECCV_2018_paper.pdf) [`seg`]
	- [Fully-Convolutional Point Networks for Large-Scale Point Clouds](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dario_Rethage_Fully-Convolutional_Point_Networks_ECCV_2018_paper.pdf) [`seg`, `captioning`; [Tensorflow](https://github.com/drethage/fully-convolutional-point-network)]
	- [PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tolga_Birdal_PPF-FoldNet_Unsupervised_Learning_ECCV_2018_paper.pdf) [`registration`; [PyTorch-unofficial](https://github.com/XuyangBai/PPF-FoldNet)]
	- [Deep Continuous Fusion for Multi-Sensor 3D Object Detection](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf) [`det`]
	- [3DFeat-Net: Weakly Supervised Local 3D
	Features for Point Cloud Registration](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf) [`match`, `registration`; [Tensorflow](https://github.com/yewzijian/3DFeatNet)]
	- [Stereo Vision-based Semantic 3D Object and Ego-motion Tracking for Autonomous Driving](http://openaccess.thecvf.com/content_ECCV_2018/papers/Peiliang_LI_Stereo_Vision-based_Semantic_ECCV_2018_paper.pdf) [`autonomous driving`]
- Others
	- [PointCNN: Convolution On X -Transformed Points](http://papers.neurips.cc/paper/7362-pointcnn-convolution-on-x-transformed-points.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/yangyanli/PointCNN); NeurIPS]
	- [Learning Representations and Generative Models for 3D Point Clouds](https://arxiv.org/pdf/1707.02392.pdf) [`autoencoder`; [Tensorflow](https://github.com/optas/latent_3d_points); ICML]
	- [RGCNN: Regularized Graph CNN for Point Cloud Segmentation](https://arxiv.org/pdf/1806.02952.pdf) [`seg`, `cls`; [Tensorflow](https://github.com/tegusi/RGCNN); ACM MM]
	- [PCN: Point Completion Network](https://arxiv.org/pdf/1808.00671.pdf) [`completion`; [Tensorflow](https://github.com/wentaoyuan/pcn); 3DV]
	- [Iterative Global Similarity Points : A robust coarse-to-fine integration solution for pairwise 3D point cloud registration](https://arxiv.org/pdf/1808.03899.pdf) [`registration`; 3DV]
	- [Semantic Classification of 3D Point Clouds with Multiscale Spherical Neighborhoods](https://arxiv.org/pdf/1808.00495.pdf) [`seg`; 3DV]
	- [Guaranteed Outlier Removal for Point Cloud Registration with Correspondences](https://www.researchgate.net/profile/Alvaro_Parra4/publication/321074557_Guaranteed_Outlier_Removal_for_Point_Cloud_Registration_with_Correspondences/links/5d2700bd92851cf440790ce5/Guaranteed-Outlier-Removal-for-Point-Cloud-Registration-with-Correspondences.pdf) [`registration`; TPAMI]
	- [Second: Sparsely embedded convolutional detection](https://www.mdpi.com/1424-8220/18/10/3337/pdf) [`det`; `Sensors`]	
	- [Rt3d: Real-time 3-d vehicle detection in lidar point cloud for autonomous driving](https://ieeexplore.ieee.org/abstract/document/#) [`det`, `autonomous driving`; IEEE Robotics and Automation Letters]
	- [HDNET: Exploiting HD Maps for 3D Object Detection](http://proceedings.mlr.press/v87/yang18b/yang18b.pdf) [`det`, `autonomous driving`; CoRL]
	- [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/pdf/1712.02294.pdf) [`det`, `autonomous driving`; IROS]
	- [Flex-Convolution(Million-Scale Point-Cloud Learning Beyond Grid-Worlds)](https://arxiv.org/pdf/1803.07289.pdf) [`cls`, `seg`; [Tensorflow](https://github.com/cgtuebingen/Flex-Convolution); ACCV]
	- [SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud](https://arxiv.org/pdf/1710.07368.pdf) [`seg`; [Tensorflow](https://github.com/BichenWuUCB/SqueezeSeg); ICRA]

- arXiv
	- [Spherical Convolutional Neural Network
	for 3D Point Clouds](https://arxiv.org/pdf/1805.07872.pdf) [`cls`]
	- [Point Convolutional Neural Networks by Extension Operators](https://arxiv.org/pdf/1803.10091.pdf) [`cls`, `seg`, `normal estimation`; [Tensorflow](https://github.com/matanatz/pcnn)]
	- [PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation](https://arxiv.org/pdf/1807.00652.pdf) [`seg`; [Tensorflow](https://github.com/MVIG-SJTU/pointSIFT)]
	- [Point Cloud GAN](https://arxiv.org/pdf/1810.05795.pdf) [`generation`; [PyTorch](https://github.com/chunliangli/Point-Cloud-GAN)]
	- [Roarnet: A robust 3d object detection based on region approximation refinement](https://arxiv.org/pdf/1811.03818.pdf) [`det`]
	- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/pdf/1803.06199) [`det`; [PyTorch](https://github.com/AI-liu/Complex-YOLO)]
	- [Classification of Point Cloud Scenes with Multiscale Voxel Deep Network](https://arxiv.org/pdf/1804.03583.pdf) [`seg`]

## 2017
- CVPR
	- [SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yi_SyncSpecCNN_Synchronized_Spectral_CVPR_2017_paper.pdf) [`seg`, `keypoints`; [Github](https://github.com/ericyi/SyncSpecCNN)]
	- [A Point Set Generation Network for 3D Object Reconstruction From a Single Image](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fan_A_Point_Set_CVPR_2017_paper.pdf) [`reconstruction`; [Tensorflow](https://github.com/fanhqme/PointSetGeneration)]
	- [Multi-View 3D Object Detection Network for Autonomous Driving](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf) [`det`, `autonomous driving`; [Tensorflow](https://github.com/bostondiditeam/MV3D)]
	- [Deep MANTA: A Coarse-To-Fine Many-Task Network for Joint 2D and 3D Vehicle Analysis From Monocular Image](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chabot_Deep_MANTA_A_CVPR_2017_paper.pdf) [`autonomous driving`]
	- **[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)** [`cls`, `seg`; [Tensorflow](https://github.com/charlesq34/pointnet)]
	- [3D Bounding Box Estimation Using Deep Learning and Geometry](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mousavian_3D_Bounding_Box_CVPR_2017_paper.pdf) [`det`]
	- [OctNet: Learning Deep 3D Representations at High Resolutions](http://openaccess.thecvf.com/content_cvpr_2017/papers/Riegler_OctNet_Learning_Deep_CVPR_2017_paper.pdf) [`cls`, `seg`, `orientation estimation`; [PyTorch](https://github.com/griegler/octnet)]
	- [3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf) [`match`, `registration`; [project](http://3dmatch.cs.princeton.edu/)]
	- [3D Point Cloud Registration for Localization using a Deep Neural Network Auto-Encoder](https://openaccess.thecvf.com/content_cvpr_2017/papers/Elbaz_3D_Point_Cloud_CVPR_2017_paper.pdf) [`registration`; [github](https://github.com/gilbaz/LORAX)]
- ICCV
	- [Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models](https://arxiv.org/pdf/1704.01222.pdf) [`cls`, `retrieval`, `seg`; [PyTorch-unofficial](https://github.com/fxia22/kdnet.pytorch)]
	- [Learning Compact Geometric Features](https://openaccess.thecvf.com/content_ICCV_2017/papers/Khoury_Learning_Compact_Geometric_ICCV_2017_paper.pdf) [`registration`; [Github](https://github.com/marckhoury/CGF)]
	- [Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models](https://openaccess.thecvf.com/content_ICCV_2017/papers/Klokov_Escape_From_Cells_ICCV_2017_paper.pdf) [`cls`, `seg`, `retrieval`]
	- [2D-Driven 3D Object Detection in RGB-D Images](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lahoud_2D-Driven_3D_Object_ICCV_2017_paper.pdf) [`det`]

- Others
	- **[Pointnet++: Deep hierarchical feature learning on point sets in a metric space](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf)** [`cls`, `seg`; [Tensorflow](https://github.com/charlesq34/pointnet2); NIPS]
	- [Deep Sets](https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf) [[PyTorch](https://github.com/manzilzaheer/DeepSets); `cls`]
	- [3D Object Proposals Using Stereo Imagery for Accurate Object Class Detection](https://arxiv.org/pdf/1608.07711.pdf) [`det`, `autonomous driving`; TPAMI]
	- [O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis](https://arxiv.org/pdf/1712.01537.pdf) [`cls`, `retrieval`, `seg`; [Github](https://github.com/microsoft/O-CNN)]
	- [Vote3deep: Fast object detection in 3d point clouds using efficient convolutional neural networks](https://arxiv.org/pdf/1609.06666.pdf) [`det`; ICRA]
	- [3d fully convolutional network for vehicle detection in point cloud](https://arxiv.org/pdf/1611.08069.pdf) [`det`; IROS; [Tensorflow](https://github.com/yukitsuji/3D_CNN_tensorflow)]
	- [SEGCloud: Semantic Segmentation of 3D Point Clouds](https://arxiv.org/pdf/1710.07563.pdf) [`seg`; 3DV]

## Before 2016 
- 2016
	- [Fast Global Registration](https://www.researchgate.net/profile/Vladlen_Koltun/publication/305983982_Fast_Global_Registration/links/57a8086908aefe6167bc8366/Fast-Global-Registration.pdf) [`registration`; ECCV; [Github](https://github.com/intel-isl/FastGlobalRegistration)]
	- [Monocular 3D Object Detection for Autonomous Driving](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Chen_Monocular_3D_Object_CVPR_2016_paper.pdf) [CVPR]
	- [Volumetric and Multi-View CNNs for Object Classification on 3D Data](http://openaccess.thecvf.com/content_cvpr_2016/papers/Qi_Volumetric_and_Multi-View_CVPR_2016_paper.pdf) [CVPR]
	- [Three-Dimensional Object Detection and Layout Prediction Using Clouds of Oriented Gradients](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Ren_Three-Dimensional_Object_Detection_CVPR_2016_paper.pdf) [CVPR]
	- [Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Sliding_Shapes_CVPR_2016_paper.pdf) [CVPR]				
	- [Fpnn: Field probing neural networks for 3d data](http://papers.nips.cc/paper/6416-fpnn-field-probing-neural-networks-for-3d-data.pdf) [NIPS]
	- [Vehicle Detection from 3D Lidar Using Fully Convolutional Network](https://arxiv.org/pdf/1608.07916) [RSS]
- 2015
	- [Robust Reconstruction of Indoor Scenes](https://www.researchgate.net/profile/Vladlen_Koltun/publication/279751165_Robust_Reconstruction_of_Indoor_Scenes/links/5599867708ae5d8f393633dc/Robust-Reconstruction-of-Indoor-Scenes.pdf) [`reconstruction`; CVPR]
	- [Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration](https://arxiv.org/pdf/1605.03344.pdf) [`registration`; TPAMI; [Github](https://github.com/yangjiaolong/Go-ICP)] 
	- [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf) [CVPR]
	- [SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_SUN_RGB-D_A_2015_CVPR_paper.pdf) [CVPR]
	- [Data-Driven 3D Voxel Patterns for Object Category Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiang_Data-Driven_3D_Voxel_2015_CVPR_paper.pdf) [CVPR]
	- [Multi-view convolutional neural networks for 3d shape recognition](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf) [ICCV]
	- [3d object proposals for accurate object class detection](http://papers.nips.cc/paper/5644-3d-object-proposals-for-accurate-object-class-detection.pdf) [NIPS]
	- [Voting for Voting in Online Point Cloud Object](https://pdfs.semanticscholar.org/521f/688f1fadaaff81269040f13270e8ec1a14d4.pdf) [RSS]
	- [Voxnet: A 3d convolutional neural network for real-time object recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf) [IROS]
- 2014
	- [Are Cars Just 3D Boxes? - Jointly Estimating the 3D Shape of Multiple Objects](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Zia_Are_Cars_Just_2014_CVPR_paper.pdf) [CVPR]
	- [Sliding shapes for 3d object detection in depth images](https://pdfs.semanticscholar.org/832e/eaede4f697f005b007798fe0d04829f656b7.pdf) [ECCV]

- 2012
	- [Are we ready for autonomous driving? the kitti vision benchmark suite](http://www.webmail.cvlibs.net/publications/Geiger2012CVPR.pdf) [CVPR]

- 2009 
	- [Fast point feature histograms (FPFH) for 3D registration](https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf) [`registration`; ICRA]
	- [Generalized-ICP](http://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf) [`registration`; RSS]
- 1992
	- [A method for registration of 3-D shapes ](https://www.researchgate.net/publication/3191994_A_method_for_registration_of_3-D_shapes_IEEE_Trans_Pattern_Anal_Mach_Intell) [`registration`; TPAMI]
- 1987
	- [Least-squares fitting of two 3-D point sets](https://www.researchgate.net/publication/224378053_Least-squares_fitting_of_two_3-D_point_sets_IEEE_T_Pattern_Anal) [`registration`; TPAMI]

## Resources
- [https://github.com/Yochengliu/awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis)
- [https://github.com/yinyunie/3D-Shape-Analysis-Paper-List](https://github.com/yinyunie/3D-Shape-Analysis-Paper-List)
- [https://github.com/NUAAXQ/awesome-point-cloud-analysis-2020](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2020)
- [https://github.com/QingyongHu/SoTA-Point-Cloud](https://github.com/QingyongHu/SoTA-Point-Cloud)
- [https://github.com/timzhang642/3D-Machine-Learning](https://github.com/timzhang642/3D-Machine-Learning)

## Tools
- [Open3D](http://www.open3d.org/): [https://github.com/intel-isl/Open3D](https://github.com/intel-isl/Open3D)
- [PCL](https://pointclouds.org/): [https://github.com/PointCloudLibrary/pcl](https://github.com/PointCloudLibrary/pcl)
- [PCL-Python](https://python-pcl-fork.readthedocs.io/en/latest/): [https://github.com/strawlab/python-pcl](https://github.com/strawlab/python-pcl)
- [Torch-Points3D](https://arxiv.org/pdf/2010.04642.pdf): [https://github.com/nicolas-chaulet/torch-points3d](https://github.com/nicolas-chaulet/torch-points3d)
- [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/): [https://github.com/open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet): [https://github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [PyTorch3D](https://arxiv.org/pdf/2007.08501.pdf): [https://github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)
- [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine): [https://github.com/NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

## Software
- [CloudCompare](https://www.danielgm.net/cc/): [https://github.com/cloudcompare/cloudcompare](https://github.com/cloudcompare/cloudcompare)
- [Semantic Segmentation Editor](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor): [https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor)
- [MeshLab](https://www.meshlab.net/): [https://github.com/cnr-isti-vclab/meshlab](https://github.com/cnr-isti-vclab/meshlab)