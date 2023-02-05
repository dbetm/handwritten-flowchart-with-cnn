# Recognition of handwritten flowcharts with CNNs
Recognition of handwritten flowcharts using convolutional neural networks to generate C source code and reconstructed digital flowcharts.

## Overview
The pipeline implemented in order to solve the problem of handwritten flowchart recognition uses image preprocessing, the input image is sent to two detectors, the shape-connector detector and the text detector. For text flow, the image is binarize and it uses [Keras OCR](https://pypi.org/project/keras-ocr/) to locate text and an implemented model with CNN + LSTM for character classifing; on the flow of shapes and connectors it uses unsharp masking and a model that is called [Faster R-CNN](https://arxiv.org/abs/1506.01497) with backbone VGG-16, which is an object detection model.

In order to augment the precision of the text detector, the technique called continual learning is used. After, some training with the text style of a specific user, the model will improve the text recognition.

Finally, the outputs are the generated source code in C, its compilation output and the digital reconstructed diagram as an image.

**Note**: Flowcharts used for testing are constructed with a defined shape-connector set. You can check it [here](https://github.com/dbetm/handwritten-flowchart-with-cnn/tree/master/model/set_shapes.png).

## Set up for testing detections
1. Create a virtual environment (venv) with Conda with name `handwritten-flowchart-recog`.
2. Download / clone this repo.
3. Acivate the new venv, move to the project directory and install the requirements: `$ pip install -r requirements.txt`
4. Shapes-connectors model:
    - Download the folder from [here](https://drive.google.com/drive/folders/1Pax_lIypAP5qYj-oDi1fFL0COUnjLe0l?usp=sharing) - I will give you access on Google Drive as soon as possible.
    - Paste it (unzipped) into `model/training_results/` (path inside the repo), so must be `model/training_results/9`
5. Text model:
    - Download IAM dataset from [here](https://drive.google.com/file/d/1gOb-bL52leremC7_OTN-qcpcwWW0ut3d/view?usp=sharing).
        - Inside `text_model`, please create a folder with name `data_model`.
        - `iam.hdf5` (94.1 MB) paste into `text_model/data_model/`
    - Download pre-trained model from [here](https://drive.google.com/file/d/1JikohW11j74PhV-FhtvTY7XorLCFUWhN/view?usp=sharing).
        - `checkpoint_weights.hdf5` (38.5 MB) paste into `text_model/output/iam/puigcerver/`

## Usage
1. Please, activate your Conda enviroment. 
2. Move to inside repository folder, example: `$ cd handwritten-flowchart-with-cnn`
3. Type: ```$ python3 handler.py ```
4. Use "Recognize flowchart" option to process a handwritten flowchart.

### Some examples of the results
![example 1](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/Images/some_results/fibo.png "Fibonacci sequence")

Calculate the nth term of the Fibonacci sequence.

![example 2](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/Images/some_results/hello_world.png "Hello world")

Hello world.

------

## Paper
A paper was written in 2022 and published on International Journal of Computer Applications, you can find it here: [Recognition of Handwritten Flowcharts using Convolutional Neural Networks](https://www.ijcaonline.org/archives/volume184/number1/32301-2022921969)

------

## Extra
Would you like to download the training dataset? [Link](https://www.kaggle.com/davbetm/flowchart-3b)

Please cite the dataset with:

- Author: ISC UPIIZ students
- Title: Flowchart 3b
- Version: 3.0
- Date: May 2020.
- Editors: Onder F. Campos and David Betancourt.
- Publisher Location: Zacatecas, Mexico.
- Electronic Retrieval Location: https://www.kaggle.com/davbetm/flowchart-3b




