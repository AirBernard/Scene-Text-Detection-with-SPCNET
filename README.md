# Scene-Text-Detection-with-SPCNET
Unofficial repository for [Scene Text Detection with Supervised Pyramid Context Network][https://arxiv.org/abs/1811.08605] with tensorflow.
## 参考代码
网络实现主要借鉴Keras版本的[Mask-RCNN](https://github.com/matterport/Mask_RCNN.git),训练数据接口参考了[argman/EAST](https://github.com/argman/EAST).论文作者在知乎的文章介绍[SPCNet](https://zhuanlan.zhihu.com/p/51397423).
## 训练
### 1、训练数据准备
训练数据放在data/下，训练数据准备在data/icdar.py：
>data
>>icdar2017
>>>Annotaions  //image_1.txt
>>>JPEGImages  //image_1.jpg
>>>train.txt   //存储训练图片的名称，例如：image_1
### 2、参数修改
修改./train.py中的学习率、batch、模型存储路径等参数，如果需要调整网络参数，在nets/config.py中修改。
### 3、执行训练
python train.py
> 代码运行环境：Python2.7 tensorflow-gpu1.13 单张1080Ti
## 测试
修改demo.py中的模型文件夹路径、测试图片路径，然后执行python demo.py
## 值得注意的地方
### 1、global text segmentation（gts）的训练
计算gts训练时损失函数时，我采用的方法是将feature pyramid的各个level产生的gts分别与全局mask gt计算softmax loss,然后取平均作为Loss_gts。因为没找到与原文关于这一块的描述，因此可能是其他的计算方法：每个level准备不同的mask_gt、将多个level的gts预测融合计算loss等等。感兴趣的可以去问问作者或者自己试试。
### 2、实现Rescore 时gts的选取
计算predict box对应的pyramid level,然后选取对应的gts计算。还有一种思路是：融合P2,P3,P4,P5的gts，然后计算box rescore.
### 3、Bounding Box的生成
MASK RCNN中是先对输出的box进行阈值过滤以及NMS，然后将剩余的回归之后的box对应的rois送入mask branch计算mask，目的是减少计算量同时获得更准确的mask。SPCNet为了减小FP与FN,对Inference流程做了修改：先对模型输出的box与mask进行Rescore,然后经过threshold filter，再对剩下的mask求Bounding Box,然后利用Poly NMS减少重叠，输出剩下的。
> 在目前代码（nets/models.py utils.py）里：是先对模型输出的box与mask进行Rescore,然后经过threshold filter与NMS，再对剩下的mask求Bounding Box,然后直接输出。
