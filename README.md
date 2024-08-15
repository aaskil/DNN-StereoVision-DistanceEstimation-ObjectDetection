# Deep Neural Networks and Stereo Vision for Enhanced Distance Estimation and Object Detectio

# Requirements
- project was ran on python 3.11 (python 3.10 to 3.12 should probably work as well)
- install pytorch based upon your enviroment from [pytorch.org](https://pytorch.org)
- install requirements from requirements.txt
```bash
pip install -r requirements.txt
```

# Project description 
This project explores the application of stereo vision combined with deep neural networks for depth estimation and instance segmentation in industrial bin picking. Conducted in collaboration with SCAPE Technologies, it involves creating a dataset of stereo image pairs and corresponding depth information obtained from a 3D scanner. We employed state-of-the-art models, such as CGI-stereo and YOLO versions 8 and 9, to develop a system capable of accurate distance estimation and object detection with segmentation. Results from our proof-of-concept study demonstrate that stereo vision systems enhanced by deep learning can achieve comparable accuracy and processing speed to LiDAR systems. While our findings confirm the robustness and efficiency of stereo vision systems, they also highlight distinct strengths and weaknesses compared to traditional methods. This research suggests a potential shift towards using deep neural networks to address current limitations in industrial bin picking.



# Research used in this project
## Alfather
[Marr, D. and Poggio, T. (1976) ‘Cooperative computation of stereo disparity’](https://apps.dtic.mil/sti/tr/pdf/ADA030748.pdf)

<!-- ---------------------------------------------------------------------- -->

## Disparity models
litterature review <br>
[Lahiri, S., Ren, J. and Lin, X. (2024) ‘Deep Learning-Based Stereopsis and Monocular Depth Estimation Techniques : A Review’, pp. 305–351.](https://www.mdpi.com/2624-8921/6/1/13)

*MobileNetV2* <br>
[Sandler, M. et al. (2018) ‘MobileNetV2: Inverted Residuals and Linear Bottlenecks’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/1801.04381)

*MC-CNN 2015 (maybe first deep learning solution):<br>
[Žbontar, J. and LeCun, Y. (2015) ‘Computing the stereo matching cost with a convolutional neural network’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/1409.4326)

MC-CNN-acrt/fast: comparison of different stereo matching algorithms, great explanations:<br>
[Žbontar, J. and LeCun, Y. (2016) ‘Stereo matching by training a convolutional neural network to compare image patches’, Journal of Machine Learning Research, 17.](https://arxiv.org/abs/1510.05970)

Content-CNN 2016, #253:<br>
[Luo, W., Schwing, A.G. and Urtasun, R. (2016) ‘Efficient deep learning for stereo matching’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf)

GC-Net 2017:<br>
[Kendall, A. et al. (2017) ‘End-to-End Learning of Geometry and Context for Deep Stereo Regression’, Proceedings of the IEEE International Conference on Computer Vision, 2017-Octob, pp. 66–75.](https://arxiv.org/abs/1703.04309)

PSMnet 2018 w/github:<br>
[Chang, J.R. and Chen, Y.S. (2018) ‘Pyramid Stereo Matching Network’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/1803.08669)
[- PSMnet github](https://github.com/JiaRenChang/PSMNet/tree/master)

GA-Net 2019:<br>
[Zhang, F. et al. (2019) ‘GA-net: Guided aggregation net for end-to-end stereo matching’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition](https://arxiv.org/pdf/1904.06587.pdf)

GWC-Net 2019:<br>
[Guo, X. et al. (2019) ‘Group-wise correlation stereo network’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/1903.04025)

ACVNet 2022:<br>
[Xu, G. et al. (2022) ‘Attention Concatenation Volume for Accurate and Efficient Stereo Matching’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/2203.02146)

CGI-Stereo 2023:<br>
[Xu, G., Zhou, H. and Yang, X. (2023) ‘CGI-Stereo: Accurate and Real-Time Stereo Matching via Context and Geometry Interaction’.](https://arxiv.org/abs/2301.02789)

<!-- ---------------------------------------------------------------------- -->

## Instance segmentation papers
Litterature review <br>
[Hafiz, A.M. and Bhat, G.M. (2020) ‘A survey on instance segmentation: state of the art’, International Journal of Multimedia Information Retrieval, 9(3).](https://arxiv.org/abs/2007.00047)

GFL <br>
[Li, X. et al. (2020) ‘Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection’, in Advances in Neural Information Processing Systems.](https://arxiv.org/abs/2006.04388)

CIoU <br>
[Zheng, Z. et al. (2022) ‘Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation’, IEEE Transactions on Cybernetics, 52(8).](https://arxiv.org/abs/2005.03572)

FCNN: <br>
[Shelhamer, E., Long, J. and Darrell, T. (2015) ‘Fully Convolutional Networks for Semantic Segmentation’, IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(4).](https://arxiv.org/abs/1411.4038)

YOLOv1
[Redmon, J. et al. (2016) ‘You only look once: Unified, real-time object detection’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/1506.02640)

Faster R-CNN: <br>
[Ren, S. et al. (2015) ‘Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks’, IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(6), pp. 1137–1149.](https://arxiv.org/abs/1504.08083)

Mask R-CNN: <br>
[He, K. et al. (2017) ‘Mask R-CNN’, IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2).](https://arxiv.org/abs/1703.06870)

SOLOv2 <br>
[Wang, X. et al. (2020) ‘SOLOv2: Dynamic and fast instance segmentation’, in Advances in Neural Information Processing Systems.](https://arxiv.org/abs/2003.10152)

ViT-Adapter: <br>
[Chen, Z. et al. (2022) ‘VISION TRANSFORMER ADAPTER FOR DENSE PREDICTIONS’.](https://arxiv.org/abs/2205.08534v4) <br>
[ViT-Adapter github](https://github.com/czczup/ViT-Adapter)

Segment Anything: <br>
[Kirillov, A. et al. (2023) ‘Segment Anything’. Available at: http://arxiv.org/abs/2304.02643.](https://arxiv.org/pdf/2304.02643.pdf)<br>
[- Segment Anything github](https://github.com/facebookresearch/segment-anything)

EVA: <br>
[Fang, Y. et al. (2023) ‘EVA: Exploring the Limits of Masked Visual Representation Learning at Scale’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/2211.07636)

YOLOv9: <br>
[Wang, C.-Y., Yeh, I.-H. and Liao, H.-Y.M. (2024) ‘YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information’.](https://arxiv.org/pdf/2402.13616v1.pdf)

<!-- ---------------------------------------------------------------------- -->

## Instance segmentation case specific
[Hoang, H.H. and Tran, B.L. (2021) ‘Accurate instance-based segmentation for boundary detection in robot grasping application’, Applied Sciences (Switzerland), 11(9).](https://www.mdpi.com/2076-3417/11/9/4248)

[Feng, Y. et al. (2022) ‘Towards Robust Part-aware Instance Segmentation for Industrial Bin Picking’, in Proceedings - IEEE International Conference on Robotics and Automation.](https://arxiv.org/abs/2203.02767)


## Domain/Case specific papers on bin picking traditional and dnn methods
[Radhakrishnamurthy, H.C. et al. (2007) ‘Stereo vision system for a bin picking adept robot’, Malaysian Journal of Computer Science, 20(1).](http://ojie.um.edu.my/index.php/MJCS/article/view/6300)

[Khalid, M.U. et al. (2019) ‘Deep workpiece region segmentation for bin picking’, in IEEE International Conference on Automation Science and Engineering.](https://arxiv.org/pdf/1909.03462.pdf)

[Chin, R.T. and Dyer, C.R. (1986) ‘Model-Based Recognition in Robot Vision’, ACM Computing Surveys (CSUR), 18(1).](https://dl.acm.org/doi/pdf/10.1145/6462.6464)

[Dolezel, P. et al. (2019) ‘Bin picking success rate depending on sensor sensitivity’, in Proceedings of the 2019 20th International Carpathian Control Conference, ICCC 2019.](https://dk.upce.cz/bitstream/handle/10195/75126/dolezel.pdf?sequence=1)

[Haleem, A. et al. (2022) ‘Exploring the potential of 3D scanning in Industry 4.0: An overview’, International Journal of Cognitive Computing in Engineering, 3.](https://www.sciencedirect.com/science/article/pii/S2666307422000171#sec0013)

[Yoshizawa, M., Motegi, K. and Shiraishi, Y. (2023) ‘A Deep Learning-Enhanced Stereo Matching Method and Its Application to Bin Picking Problems Involving Tiny Cubic Workpieces’, Electronics (Switzerland), 12(18).](https://www.mdpi.com/2079-9292/12/18/3978)

<!-- ---------------------------------------------------------------------- -->

## 3D-2D extrinsic calibration
[Luo, Z., Yan, G. and Li, Y. (2023) ‘Calib-Anything: Zero-training LiDAR-Camera Extrinsic Calibration Method Using Segment Anything’.](https://arxiv.org/abs/2306.02656v1)

[Ma, T. et al. (no date) ‘CRLF: Automatic Calibration and Refinement based on Line Feature for LiDAR and Camera in Road Scenes’.](https://arxiv.org/abs/2103.04558)

[joint-camera-intrinsic-and-lidar-camera](https://cs.paperswithcode.com/paper/joint-camera-intrinsic-and-lidar-camera)<br>

[OpenCalib: A Multi-sensor Calibration Toolbox for Autonomous Driving](https://paperswithcode.com/paper/opencalib-a-multi-sensor-calibration-toolbox)<br>

<!-- ---------------------------------------------------------------------- -->

## Dataset papers
KITTI 2012:<br>
[A., Lenz, P. and Urtasun, R. (2012) ‘Are we ready for autonomous driving? the KITTI vision benchmark suite’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf)

Middlebury dataset:<br>
[Scharstein, D. et al. (2014) ‘High-resolution stereo datasets with subpixel-accurate ground truth’, in Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics).](https://elib.dlr.de/90624/1/ScharsteinEtal2014.pdf)

KITTI 2015:<br>
[Menze, M. and Geiger, A. (2015) ‘Object scene flow for autonomous vehicles’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://www.cvlibs.net/publications/Menze2015CVPR.pdf)

SceneFLow: <br>
[Mayer, N. et al. (2016) ‘A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation’, in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/1512.02134)

<!-- ---------------------------------------------------------------------- -->

## Depth Completion
[Khan, M.A.U. et al. (2022) ‘A Comprehensive Survey of Depth Completion Approaches’, Sensors.](https://www.mdpi.com/1424-8220/22/18/6969)

[Tang, J. et al. (2024) ‘Bilateral Propagation Network for Depth Completion’.](https://arxiv.org/abs/2403.11270)

[Uhrig, J. et al. (2018) ‘Sparsity Invariant CNNs’, in Proceedings - 2017 International Conference on 3D Vision, 3DV 2017.](https://arxiv.org/abs/1708.06500)
