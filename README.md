# tensorflow_object_detection_api_demo
```


pip install --upgrade tensorflow


wget  https://github.com/tensorflow/models/archive/dcfe009a024854207c9067d785c105f5ebf5a01b.zip
unzip dcfe009a024854207c9067d785c105f5ebf5a01b.zip
mv models-dcfe009a024854207c9067d785c105f5ebf5a01b models

pip install Cython
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib

cd /output/models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py

cd /output
git clone https://github.com/sanfooh/tensorflow_object_detection_api_demo.git

cd /output/tensorflow_object_detection_api_demo

wget  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar -xzvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz


python create_pet_tf_record.py

mkdir mytrain

python /output/models/research/object_detection/train.py --train_dir=mytrain/ --pipeline_config_path=net.config --logtostderr


```
