# Hand gesture detector


## Running CNN file

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

### Running pro.py

#### Install the required packages
```bash
conda install -c conda-forge keras
conda install -c anaconda numpy
```
Set the training data and testing data in the same directory where code is present
Use the below code to get the value for each class(label)
```bash
training.class_indices
```

### To run final.py
We need to run final.py which is present in models/research/object_detection/
We need to install these packages before running the code
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

1)Open anaconda
2)Open Spyder
3)Make sure you are in right path where the code is present
4)Change the image name you want to test
5)Now the file
6)Get the output image with predicted gesture name.

## To train your own model
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
Gather some of the images where hands are present and seperate 80% of images to training folder and 20% of images to testing folder. These folders are present in the images folder in object detection.
The more the images the better the output
Get the labellmg from the "required files" which is in the zip file

After running the labellmg.exe file open the directory containing all the images where hands are present
Use create box and make a box around the hand
Label the bax as hand
Do this both for training and testing images
Now we generated xml file for each image.

### Convert the xml data to csv file
```bash
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```
Make sure you are in object detection folder.
This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.
In generate tfrecord.py file make the changes according to your label
```bash
def class_text_to_int(row_label):
    if row_label == 'hand':
        return 1
    else:
        None
```
In my case I am only predicting hand

```bash
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.<br />
Make changes in the labelmap.pbtxt file which is present in traing folder

```bash
item {
  id: 1
  name: 'hand'
}
```

In my case I am predicting only hand

### Train the model
```bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

### After training, get the frozen inference graph
```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
#### Make changes in the detection file by making sure the inference graph, image, frazen inference graph and labelmap.pbtxt are in right path.
