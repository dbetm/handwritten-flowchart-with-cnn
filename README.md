# Recognition of handwritten flowcharts with CNNs
Recognition of handwritten flowcharts using convolutional neural networks to generate C source code and reconstructed digital flowchart.

## General description
The pipeline implemented in this project in order to solve the problem of recognition of handwritten flowcharts uses preprocessing of images, the image is send to two detectors, the shapes and connectors detector and the text detector. In flow for text, the image is binarize and uses [Keras OCR](https://pypi.org/project/keras-ocr/) for locate text and an implemented model with CNN + LSTM for classifing characters; besides, the flow to shapes and connectors uses masking unsharp and an implemented model that's called [Faster R-CNN](https://arxiv.org/abs/1506.01497) with backbone VGG-16. 

In order to augment text detector precision the technique called continual learning  is used. After, some training with the texts of a certain user, the model will improve.

Finally, like output is generated source code in C and its compilation output, and the reconstructed diagram in image digital.

Note: The flowcharts used for testing are constructed with a defined set of shapes and connectors. You can see [here](https://github.com/dbetm/handwritten-flowchart-with-cnn/tree/master/model/set_shapes.png).

## How to set up for testing detections
1. Create a virtual enviroment with Conda with name 'tt', the list of requirements is [here](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/requirements.txt).
2. Download this repo.
3. Model for shapes and connectors:
    - Download the folder [here](https://drive.google.com/drive/folders/1Pax_lIypAP5qYj-oDi1fFL0COUnjLe0l?usp=sharing).
    - Paste it (unzipped) into model/training_results/ (path inside the repo)
4. For text.
    - Download IAM dataset [here](https://drive.google.com/file/d/1gOb-bL52leremC7_OTN-qcpcwWW0ut3d/view?usp=sharing)
        - Inside text_model create a folder with name: 'data_model'.
        - iam.hdf5 (94.1 MB) paste into text_model/data_model/
    - Download trained model [here](https://drive.google.com/file/d/1JikohW11j74PhV-FhtvTY7XorLCFUWhN/view?usp=sharing)
        - checkpoint_weights.hdf5 (38.5 MB) paste into text_model/output/iam/puigcerver/

## Usage
1. Activate your Conda enviroment.
2. Change terminal to repository folder.
3. Type: ```$ python3 handler.py ```
4. Use "Recognize flowchart" option for testing detections with handwritten flowcharts.

### Some examples of the results
![example 1](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/Images/some_results/fibo.png "Fibonacci sequence")

Calculate the nth term of the Fibonacci sequence, compilation output: None.

![example 2](https://github.com/dbetm/handwritten-flowchart-with-cnn/blob/master/Images/some_results/hello_world.png "Hello world")

Hello world, compilation output: None.


