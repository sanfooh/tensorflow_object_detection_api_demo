# tensorflow_object_detection_api_demo
```
#在根目录下创建一个output目录
mkdir /output
cd /output/

#下载旧版本的tensorflow model（object api 包含在model里）,最新版本的api存在问题（当前2018.4.20）
wget  https://github.com/tensorflow/models/archive/dcfe009a024854207c9067d785c105f5ebf5a01b.zip
unzip dcfe009a024854207c9067d785c105f5ebf5a01b.zip 
mv models-dcfe009a024854207c9067d785c105f5ebf5a01b models
rm dcfe009a024854207c9067d785c105f5ebf5a01b.zip 

#安装依赖项
pip install Cython
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib

#安装object detection api 并验证
cd /output/models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py

#下载图片及标注文件
cd /output
git clone https://github.com/sanfooh/tensorflow_object_detection_api_demo.git

#下载预训练文件
cd /output/tensorflow_object_detection_api_demo
wget  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar -xzvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
rm -r ssd_mobilenet_v1_coco_2017_11_17.tar.gz

#生成tfrecord数据集
python create_tf_record.py 

#可使用check_tfrecord.ipynb验证一下tfrecord是否有问题


#开始训练
mkdir mytrain
python /output/models/research/object_detection/train.py --train_dir=mytrain/ --pipeline_config_path=net.config --logtostderr

#生成发布模型，注意要根据mytrain文件夹下实际情况，修改下面的“4701”
python /output/models/research/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path /output/tensorflow_object_detection_api_demo/net.config \
--trained_checkpoint_prefix /output/tensorflow_object_detection_api_demo/mytrain/model.ckpt-35187 \
--output_directory /output/tensorflow_object_detection_api_demo/mymodel

#使用inference.ipynb来推理


引用  
https://www.jianshu.com/p/4ec080f709d8  
https://github.com/priya-dwivedi/Deep-Learning/tree/master/tensorflow_toy_detector  

```
