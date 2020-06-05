# Hand gesture detector


## Running CNN file(gesture.py)

Download and install Anaconda [Anaconda](https://www.anaconda.com/products/individual)

#### Install the required packages
```bash
conda install -c conda-forge keras
conda install -c anaconda numpy
```
You can directly install packages in Anaconda Packages
1. Goto environments
2. Select environment to install
3. Search in packages
4. Install

Make sure you are in the right directory other wise change the path of training and testing datasets in the code

### Running simple_detect.py

Get the input image from sample images
#### Install the required packages
```bash
conda install -c conda-forge keras
conda install -c anaconda numpy
```

Use the below code to get the value for each class(label)
```bash
training.class_indices
```

### To train your own model
Using the below method we made a model using tensorflow which detect's hand in the given image.

Download and install Anaconda [Anaconda](https://www.anaconda.com/products/individual)
Create new environment with python 3.5 or 3.6
##### Create Virtual Environment
```bash
conda create --name myenv
conda create -n tensorflow1 python=3.6
```
#### Activate the model and install pip
```bash
C:\> activate tensorflow1
(tensorflow1) C:\>python -m pip install --upgrade pip
```
#### Install Tensorflow 
Make sure to install version 1 tensorflow
```bash
pip install --ignore-installed --upgrade tensorflow==1.4
```
#### Install Some Packages where we use it in code
```bash
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
```

Check the models folder in the zip file
Set the python path
Change according to your models path where you saved in your system
```bash
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```

Change the directory to the research directory in models
```bash
(tensorflow1) C:\> cd C:\tensorflow1\models\research
```
And copy paste it in the conda prompt
```bash
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
These are proto files which are used in the code

Run the following commands
```bash
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

Test whether your object detection works
```bash
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
It will open a jupyter notebook where you run your code and the output will be a predicted dog image.

If this works then setup is ready and we can make our own predictor.

### Make the training and testing dataset
Gather some of the images where hands are present
The more the images the better the output
Get the labellmg from the "required files" which is in the zip file

