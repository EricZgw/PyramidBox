PyramidBox
===
This is an unofficial Tensorflow re-implementation of [PyramidBox: A Context-assisted Single Shot Face Detector](https://arxiv.org/abs/1803.07737?context=cs), which achieves superior performance among the state-of-the-art on the two common face detection benchmarks, FDDB and WIDER FACE.

## Note
There is still a gap in performance from the paper. May be caused by several reasons:
* Without implementing data-anchor-sampling.
* Differences of data augment from original.
* The batch size in the paper is 16, but I used 1 because of the limitation of memory.
* Differences of deep learning framework.

## Results
### Face Detection
![](https://github.com/guodongxiaren/ImageCache/raw/master/Logo/foryou.gif)
![](https://github.com/guodongxiaren/ImageCache/raw/master/Logo/foryou.gif)

Results on WIDER FACE validation set
<center>

| Method | AP Easy | AP Medium | AP Hard |
|:-------|:-------:|:-------:|:-------:
| original | 96.1 | 95.0 | 88.9 |
| **this repo** | **90.6** | **88.8** | **73.4** |

</center>

## Usage
### Prerequisites
(Only tested on) Ubuntu 16.04 with:
* Python 3.5
* Tensorflow-gpu 1.4
* OpenCV3
### Clone the repo
```
git clone https://github.com/EricZgw/PyramidBox.git
```
Download PyramidBox models form [BaiduYun](https://pan.baidu.com/s/1kC-G_e8louDig5Y-NK142g) .
### Demo
Run the following script for visualization:
```
python demo.py
```
### Train on WIDER FACE Datasets
1. Download pre-trained VGG16 models from [here](https://github.com/tensorflow/models/tree/master/research/slim) and put it to /checkpoints. <br>
2. Download [WIDER FACE Datasets](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and convert to VOC format. Path looks like below:
```
datasets/
       |->widerface/
       |    |->WIDER_train/
       |    |->WIDER_val/
       |    |->WIDER_test/
       |    |->Annotations/
       |    |->JPEGImages/
       |    |...
```
3. Run the following script to generate TFRecords:
```
cd datasets
python pascalvoc_to_tfrecords.py
```
You can run `check_data_io.py` to check data. This step is not necessary.<br>
4. The training strategy is two-stages:<br> 
First run `train_model.py` with below to train additional PyramidBox layers:
```
self.fine_tune_vgg16 = False
```
Then set `self.fine_tune_vgg16 =Ture` to run `train_model.py` to train total network.
### Validation
Run the following script for evaluation and get mAP:
```
python widerface_eval.py
cd eval/eval_tools
octave wider_eval.m
```

## TODO
* Add data-anchor-sampling
* Try more logical and rigorous data augment
* Transfer to other backbone networks
Please stay tuned.

## Reference

[SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)<br>
[SSD_tensorflow_VOC](https://github.com/LevinJ/SSD_tensorflow_VOC)

## Contact
If you find any problems, welcome to open a new issue or contact zhaogw93@126.com .
