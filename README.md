# Recognition of handwritten flowcharts with CNNs
Recognition of handwritten flowcharts using convolutional neural networks to generate C source code and reconstructed digital flowchart.

## Overview
The pipeline implemented in order to solve the problem of recognition of handwritten flowcharts uses image preprocessing, the input image is sending to two detectors, the shape-connector detector and to the text detector. For text flow, the image is binarize and it uses [Keras OCR](https://pypi.org/project/keras-ocr/) for locate text and an implemented model with CNN + LSTM for character classifing; moreover, on the flow for shapes and connectors it uses unsharp masking and a model that is called [Faster R-CNN](https://arxiv.org/abs/1506.01497) with backbone VGG-16. 

In order to augment the precision of the text detector, the technique called continual learning is used. After, some training with the text style of a certain user, the model will improve the text recognition.

Finally, the outputs are the generated source code in C, its compilation output and the digital reconstructed diagram (image format).

Note: Flowcharts used for testing are constructed with a defined shape set and connectors. You can see [here](https://github.com/dbetm/handwritten-flowchart-with-cnn/tree/master/model/set_shapes.png).

## How to set up for testing detections
1. Create a virtual enviroment with Conda with name 'tt', the requirement list is [here](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/requirements.txt).
2. Download / clone this repo.
3. Shapes-connectors model:
    - Download the folder [here](https://drive.google.com/drive/folders/1Pax_lIypAP5qYj-oDi1fFL0COUnjLe0l?usp=sharing).
    - Paste it (unzipped) into `model/training_results/` (path inside the repo), so must be `model/training_results/9`
4. Text model:
    - Download IAM dataset [here](https://drive.google.com/file/d/1gOb-bL52leremC7_OTN-qcpcwWW0ut3d/view?usp=sharing)
        - Inside `text_model`, please create a folder with name 'data_model'.
        - `iam.hdf5` (94.1 MB) paste into `text_model/data_model/`
    - Download pre-trained model [here](https://drive.google.com/file/d/1JikohW11j74PhV-FhtvTY7XorLCFUWhN/view?usp=sharing)
        - `checkpoint_weights.hdf5` (38.5 MB) paste into `text_model/output/iam/puigcerver/`

## Usage
1. Please, activate your Conda enviroment. 
2. Move to inside repository folder, example: `$ cd handwritten-flowchart-with-cnn`
3. Type: ```$ python3 handler.py ```
4. Use "Recognize flowchart" option to testing detections with handwritten flowcharts.

### Some examples of the results
![example 1](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/Images/some_results/fibo.png "Fibonacci sequence")

Calculate the nth term of the Fibonacci sequence, compilation output: None.

![example 2](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/Images/some_results/hello_world.png "Hello world")

Hello world, compilation output: None.

## Extra
Would you like to [download](https://www.kaggle.com/davbetm/flowchart-3b) the dataset? 

Please cite with:

- Author: ISC UPIIZ students
- Title: Flowchart 3b
- Version: 3.0
- Date: May 2020.
- Editors: Onder F. Campos and David Betancourt.
- Publisher Location: Zacatecas, Mexico.
- Electronic Retrieval Location: https://www.kaggle.com/davbetm/flowchart-3b




