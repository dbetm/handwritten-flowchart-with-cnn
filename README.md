# Recognition of handwritten flowcharts with CNNs.
Recognition of handwritten flowcharts using convolutional neural networks to generate C source code and reconstructed digital flowchart.

Next, a Brief evaluation report on the implemented system.

**Demonstration:** Generation of the digital image of the reconstructed flowchart and source code in C language.

**General aspects of the system**
* Dataset (shapes and arrows): [flowchart-3b v2](https://mega.nz/file/ce43iDSb#x4n6WiWK3HBEXdXu8YYEEObi92EwsNOlboDNrkmVEaY).
  - 13 classes, template photos of 12 handwritten symbols of each class.
* Object detection model (shapes and arrows): Faster R-CNN with VGG-16.
* Object detecion model (text): [Keras OCR](https://pypi.org/project/keras-ocr/).
* Keras with Tensorflow for the implementation of the deep learning models (convolutional neural networks).
* Graphviz for Python to generate the digital image of the diagram.
* [Requirements to install](https://drive.google.com/file/d/15uwCCZs8yFWHcfRW3c-Oh8RQjerPDWil/view?usp=sharing) the virtual environment in Conda.

**About training (model for shapes and connectors):**
- 500 epochs in a time of 6 hours.
- Software: Conda 4.7.12, Tensorflow 2.0.0, Keras 2.3.1, SO: GNU/Linux Ubuntu 18.04 LTS.
- Hardware: 16 GB RAM, 123 SSD, Intel core  i7-9700, GeForce GTX 1660.
- The best results were when using a pre-trained model to initialize the weights: [vgg16_weights_tf_dim_ordering_tf_kernels](https://github.com/fchollet/deep-learning-models/releases)

**Better results when making different variations on some hyperparameters**
id | variation | mAP | loss
------------ | ------------- | ------------ | -------------
1 | Default (view below) | 0.9852 | 0.3130
2 | Data augmentation: <br>use_horizontal_flips=True <br>use_vertical_flips=True| 0.9799 | 0.3271
3 | Anchors: <br>scales={64^2,128^2,256^2}<br>ratios={2:1,1:1,1:2} | 0.9876 | 0.3576
4 | RoIs: 15 | 0.9818 | 0.3671
5 | RPN max overlap: 0.65 | 0.9839 | 0.3790
6 | Optimizer: RMSprop | 0.9706 | 0.3088

**Default hyperparameters**
- use_horizontal_flips = False
- use_vertical_flips = False
- epoch_lenght = 32
- learning_rate = 0.00001
- anchor_box_scales = (128, 256, 512)
- anchor_box_ratios = {(1, 1), (1./math.sqrt(2), 2./math.sqrt(2)), (2./math.sqrt(2), 1./math.sqrt(2))}
- min_image_side = 600
- img_channel_mean = (103.939, 116.779, 123.68)
- img_scaling_factor = 1.0
- num_rois = 32
- rpn_stride = 16
- std_scaling = 4.0
- classifier_regr_std = (8.0, 8.0, 4.0, 4.0)
- rpn_min_overlap = 0.3
- rpn_max_overlap = 0.7
- classifier_min_overlap = 0.1
- classifier_max_overlap = 0.5
- optimizer =  Adam

**How accurate is the best?**

[Download best model here](https://drive.google.com/open?id=1vqM2mkwkp9tNKybGvLD_LxCi01MzJufx)

The result with id equal to 3, had a better performance (0.9876) on the mAP metric in the validation dataset. The following table shows the AP by class:

CLass | AP
------------ | -------------
Arrow line up | 0.9301
Arrow line down | 0.9294
Arrow line left | 0.9787
Arrow line right | 0.9765
Arrow rectangle up | 0.9890
Arrow rectangle down | 0.9990
Arrow rectangle left | 1.0
Arrow rectangle right | 1.0
Start end | 0.9999
Process | 1.0
Decision | 0.9999
Print | 0.9999
Scan | 0.9976
