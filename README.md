## Objective
Object Detection Task
1) Train a custom object detection model for the provided data  
2) Host it on tensorflow serving. 
3) Make a Rest API.

**Assumptions** -
1) Classes:  
·      ZIP : Footwear consisting of zipper in it. 
·      BUCKLE : Footwear consisting of buckle.

**Output** :
1) Json result to show location of object in image
2) Result image with highlighted area of object with probability
3) Flask application to upload image file [handled edge cases for no error]
4) Object detection model as pb format with checkpoint
5) Splitted Training and Test data created from XML

**Constraints / Notes** ::
1) Assumed there can be both zip and buckle in one image
2) Used TensorFlow object detection API - Pretrained object detection model of tensorflow is used so as to use that as checkpoint for further traing on footwear dataset. It is used as data is not much
3) Used faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28 pre trained model as it is trained on different variety of Images
4) Trained for 100 iteration accuracy can be increase if train model on cloud using GPU for more iteration. Result accuracy can be improved if train for more iteration or more data
5) Model is hosted on tensorflow serving on docker for local machine. It can be scale for more users if use kubernetes and cloud

**** 

## Tools use 
> Python 3

> Main Libraries Used -
1) tensorflow
2) pillow
3) numpy
4) flask
5) pandas

**** 

## Installing and Running

> Folders Used -
1) src - For Flask Application
2) models - Having pre trained model on tensorflow which is used for training
3) images - Images of the footwear
4) annotations - Images annotation of the footwear
5) data - having data in csv, tensorflow record format. It also has pbtxt for label mapping used


```sh
$ cd footwear_detection
$ pip install -r requirements.txt
``` 

For making dataset using XML
```sh
$ python xml_process.py
```
For converting dataset to tensorflow format having label map
```sh
$ python generate_tf_record.py --csv_input=data/train.csv  --output_path=train.record
$ python generate_tf_record.py --csv_input=data/test.csv  --output_path=test.record
```

For Training using pre trained model
```sh
cd models/object_detection
$ python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/pipeline.config
$ python legacy/eval.py --logtostderr --pipeline_config_path=training/pipeline.config --checkpoint_dir=training/ --eval_dir=eval/
```

For converting checkpint to pb model format
```sh
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=$(pwd)/models/research/object_detection/training/pipeline.config
TRAINED_CKPT_PREFIX=$(pwd)/models/research/object_detection/training/model/model.ckpt-100
EXPORT_DIR={pwd}/models/research/object_detection/train_model
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```
For Running Docker For Tensorflow Serving
From ROOT Folder
```sh
docker pull tensorflow/serving
MODEL_DIR_NAME=footwear
MODEL_NAME=footwear
docker run -t --rm -p 8501:8501 \
   -v "$(pwd)/data/$MODEL_DIR_NAME:/models/$MODEL_NAME" \
   -e MODEL_NAME=$MODEL_NAME \
   tensorflow/serving &
```
For Running Flask Application
```sh
$ python src/footwear/webserver.py
```
Use http://0.0.0.0:5000/footwear/img_upload for web application

****
## Various Steps in approach are -

1) XML processing techniques used -

    Used ElementTree of XML to get these parameters -
    filename,width,height,class,xmin,ymin,xmax,ymax
    
    **Note**:
    Detect data which don't have name in there xml during XML processing

2) Data spliting to train and test datset

3) Conversion of splited data into tensorflow record format

4) Training of data using pre trained model - faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28
It is kept in model folder in object detection folder. Result is saved in training folder. 100 Iteration is used

5) Last checkpoint output of tensorfow is converted into pb format

6) Tensorflow pb format model is hosted using tensorflow serving on Docker

7) Flask Application is made to use docker local hosted URL for requesting model
    It has following folder 
    - object_detection - For using object_detection in flask to visulaise image with predicted label
    - templates - For web application to use rest API.
        - img_upload - For Uploading Image
        - result - For Showing Result
    - flask_media - For saving image uploaded
    - data_label_path - For saving data label used
    - flask_result - For saving result