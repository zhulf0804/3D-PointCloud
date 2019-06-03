## Datasets

This is only a brief introduction to some datasets, containing ***semantic segmentation currently*** . More detailed information and more datasets will be updated in the future. 

+ [PASCAL VOC(Visual Object Classes) 2012](#voc)
+ [PASCAL-Context](#pascal_context)
+ [NYUDv2](#nyudv2)
+ [SUN-RGBD](#sun_rgbd)
+ [Microsoft COCO](#coco)
+ [Cityscape](#cityscape)
+ [CamVid](#camvid)
+ [ADE20K](#ade20k)


## <span id="voc">PASCAL VOC(Visual Object Classes) 2012</span>
+ [MainPage](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) 
+ [The Pascal Visual Object Classes Challenge: A Retrospective](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf) [IJCV 2014]
+ [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6)
+ [[download for trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)] [[download for test](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar)] [[download for aug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)]

**Overview** There are 20 object classes in the dataset.
 
| Group | Classes |
| :---: | :---: |
| Person | person |
| Animal | bird, cat, cow, dog, horse, sheep |
| Vehicle | aeroplane, bicycle, boat, bus, car, motorbike, train |
| Indoor | bottle, chair, dining table, potted plant, sofa, tv/monitor |
	
Three main object recognition competitions on this dataset: classification, detection and segmentation.
	
The dataset directory is as follows:
	
```
|--VOCdevkit
      |--VOC2012(trainval)
            |--Annotations (17,125)
                    |--*.xml
            |--ImageSets
                    |--Action (33)
                            |--*.txt
                    |--Layout
                            |--train.txt
                            |--trainval.txt
                            |--val.txt
                    |--Main   (63)
                            |--*.txt
                    |--Segmentation
                            |--train.txt (1464 lines)
                            |--trainval.txt (2913 lines)
                            |--val.txt (1449 lines)
            |--JPEGImages  (17,125)
                    |--*.jpg
            |--SegmentationClass  (2913)
                    |--*.png
            |--SegmentationObject (2913)
                    |--*.png
      |--VOC2012(test)
            |--Annotations (5,138)
                    |--*.xml
            |--ImageSets
                    |--Action (11)
                            |--*.txt
                    |--Layout
                            |--test.txt
                    |--Main   (21)
                            |--*.txt
                    |--Segmentation
                            |--test.txt (1456 lines)
            |--JPEGImages  (16,135)
                    |--*.jpg
```

## <span id="pascal_context">PASCAL-Context</span>
+ [MainPage](https://www.cs.stanford.edu/~roozbeh/pascal-context/) 
+ [The Role of Context for Object Detection and Semantic Segmentation in the Wild](https://www.cs.toronto.edu/~urtasun/publications/mottaghi_et_al_cvpr14.pdf) [CVPR 2014]
	
**Overview** This dataset is a set of additional annotations for PASCAL VOC 2010. It goes beyond the original PASCAL semantic segmentation task by providing annotations for the whole scene. Training and validation contains 10,103 images while testing contains 9.637 images. The statistics section has a full list of 400+ labels. See **pascal-voc.txt**

## <span id="nyudv2">NYUDv2</span> 
+ [MainPage](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)  
+ [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf) [ECCV 2012]

**Overview** The NYU-Depth V2 data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect. 
	
It features:

+ 1449 densely labeled pairs of aligned RGB and depth images
+ 464 new scenes taken from 3 cities
+ 407,024 new unlabeled frames
Each object is labeled with a class and an instance 	+ number (cup1, cup2, cup3, etc

The dataset has several components:

+ Labeled: A subset of the video data accompanied by dense multi-class labels. This data has also been preprocessed to fill in missing depth labels.
+ Raw: The raw rgb, depth and accelerometer data as provided by the Kinect.
+ Toolbox: Useful functions for manipulating the data and labels

	
## <span id = "sun_rgbd">SUN-RGBD</span> 
+ [MainPage](http://rgbd.cs.princeton.edu/)
+ [SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_SUN_RGB-D_A_2015_CVPR_paper.pdf) [CVPR2015]

**Overview** The dataset is captured by four different sensors and contains 10,335 RGB-D images, at a similar scale as PASCAL VOC. The whole dataset is densely annotated and includes 146,617 2D polygons and 64,595 3D bounding boxes with accurate object orientations, as well as a 3D room layout and scene category for each image. This dataset enables us to train data-hungry algorithms for scene-understanding tasks, evaluate them using meaningful 3D metrics, avoid
overfitting to a small testing set, and study cross sensor bias.

Semantic segmentation in the 2D image domain is currently the most popular task for RGB-D scene understanding. In this task, the algorithm outputs a semantic label for each pixel in the RGB-D image. We use the standard average accuracy across object categories for evaluation

## <span id="coco">Microsoft COCO</span>: 
+ [MainPage](http://cocodataset.org/)
+ [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf) [ECCV 2014]

**Overview** COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features:
+ Object segmentation
+ Recognition in context
+ Superpixel stuff segmentation
+ 330K images (>200K labeled)
+ 1.5 million object instances
+ 80 object categories
+ 91 stuff categories
+ 5 captions per image
+ 250,000 people with keypoints

## <span id = "cityscape">Cityscape</span>: 
+ [MainPage](https://www.cityscapes-dataset.com/)  
+ [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/pdf/1604.01685.pdf) [CVPR 2016]
+ [Tool](https://github.com/mcordts/cityscapesScripts)

**Overview** The Cityscapes Dataset focuses on semantic understanding of urban street scenes. 5,000 annotated images with fine annotations and 20,000 annotated images with coarse annotations are in the dataset. The dataset directory is as follows:
	
```
|--cityscape
      |--leftImg8bit_trainvaltest
                 |--leftImg8bit (5,000)
                       |--train (2975)
                       |--val   (500)
                       |--test  (1525)
      |--gtFine_trainvaltest    (5000 * 4)
                 |--gtFine
                       |--train (2975 * 4)
                       |--val   (500 * 4)
                       |--test  (1525 * 4)
      |--leftImg8bit_trainextra
                 |--leftImg8bit (20,000)
                       |--train_extra (20000)
      |--gtCoarse
                 |--gtCoarse    (23475 * 4)
                       |--train (2975 * 4)
                       |--train_extra (20000 * 4)
                       |--val   (500 * 4)                 
```
The number in parentheses indicates the number of images. The images in the annotation(gtFine_trainvaltest, gtCoarse) directory contains **4 types: color, instanceIds, labelIds and polygons**. The test set in the annotation directory doesn't have labels(ground truth).
	
It contains 30 classes object. The class definitions details can be seen in the following:
 

| Group | Classes |
| :---: | :---: |
| flat | road · sidewalk · parking+ · rail track+ |
| human | person* · rider* |
| construction | building · wall · fence · guard rail+ · bridge+ · tunnel+ |
| object | pole · pole group+ · traffic sign · traffic light |
| nature | vegetation · terrain |
| sky | sky |
| void | 	ground+ · dynamic+ · static+ |

	
## <span id = "camvid">CamVid</span>
+ [[MainPage](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)]  [[Related](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/)] 
+ [Semantic object classes in video: A high-definition ground truth database](http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf) [Pattern Recognition Letters 2009] 
+ [Segmentation and Recognition using Structure
from Motion Point Clouds](https://velodynelidar.com/lidar/hdlpressroom/pdf/papers/journal_papers/Segmentation%20and%20Recognition%20using%20Structure%20from%20Motion%20Point%20Clouds.pdf) [ECCV 2008]

**Overview** The Cambridge-driving Labeled Video Database(Cam Vid) is the first collection of videos with object class semantic labels, complete with metadata. The database provides ground truth labels that associate each pixel with one of 32 semantic classes. 
	
| Group | Classes |
| :---: | :---: |
| Moving object | Animal · Pedestrian · Child · Rolling cart/luggage/pram · Bicyclist · Motorcycle/scooter · Car (sedan/wagon) · SUV / pickup truck · Truck / bus · Train · Misc |
| Road | Road == drivable surface · Shoulder · Lane markings drivable · Non-Drivable |
| Ceiling | Sky · Tunnel · Archway |
| Fixed objects | Building · Wall · Tree · Vegetation misc. · Fence · Sidewalk · Parking block · Column/pole · Traffic cone · Bridge · Sign / symbol · Misc text · Traffic light · Other |
	
Labeled Images (***701*** so far)

***Note:*** Some links on the [MainPage](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) are invalid, including the raw images download link, ***you can visit the [Related](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/) to get the data.***

The dataset directory is as follows:

```
|--CamVid
      |--701_StillsRaw_full   (701)
              |--*.png
      |--LabeledApproved_full (701)
              |--*.png
      |--label_colors.txt (32 lines)
              |-- color(like 64, 128, 64) classes(like Animal)
```

For the train/val/test, we can refer to the [site](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid), the directory can be as follows:

```
|--CamVid
     |--train       (367)
          |--*.png
     |--trainannot  (367)
          |--*.png
     |--val         (101)
          |--*.png  
     |--valannot    (101)
          |--*.png
     |--test        (233)
          |--*.png
     |--testannot   (233)
          |--*.png
     |--train.txt   (367 lines)
     |--val.txt     (101 lines)
     |--test.txt    (233 lines)
```


## <span id = "ade20k">ADE20K</span> 
+ [MainPage](http://groups.csail.mit.edu/vision/datasets/ADE20K/) 
+ [Scene Parsing through ADE20K Dataset](http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf) [ECCV 2017]
+ [Semantic Understanding of Scenes through the ADE20K Dataset](https://arxiv.org/pdf/1608.05442.pdf) [IJCV 2019]

**Overview** The dataset is for scene parsing and semantic segmentation task. Training set contains 20,210 images, validation set contains 2,000 images, test set is to be released later 

Each folder contains images separated by scene category (same scene categories than the Places Database). For each image, the object and part segmentations are stored in two different png files. All object and part instances are annotated sparately.

For each image there are the following files:

+ *.jpg: RGB image.

+ *_seg.png: object segmentation mask. This image contains information about the object class segmentation masks and also separates each class into instances. The channels R and G encode the objects class masks. The channel B encodes the instance object masks. The function loadAde20K.m extracts both masks.

+ *_seg_parts_N.png: parts segmentation mask, where N is a number (1,2,3,...) indicating the level in the part hierarchy. 

+ *_.txt: text file describing the content of each image (describing objects and parts). This information is redundant with other files. 
	