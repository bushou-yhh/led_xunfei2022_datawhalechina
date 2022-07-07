# 科大迅飞2022 -- LED灯的色彩和均匀性检测挑战赛（DataWhale)

注：就分类任务而言，开源的工具包有很多，例如：PyTorchImageModels(timm)、飞浆的PaddleClas、openmmlab的MMclassification等，它们通常包含诸多模型的实现，而且代码的质量相对较高，使用它们的同时，阅读它们的代码，会让我们学到很多东西。本次，我们抛砖引玉，采用openmmlab的mmclassification这个工具箱作为示例，教大家如何使用他们来帮助我们快速构建我们需要的模型，完成任务。


## 第一步，安装依赖

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y  #这里根据自己电脑的环境进行安装，比如我 选择的是 pytorch 1.9.0 cudatoolkit 10.2
conda activate open-mmlab # 创建虚拟环境
pip3 install openmim #安装openmim，方便使用mmlab的全家桶，方便下载各种权重、参数文件以及后面的模型部署等等
mim install mmcv-full # 安装mmcv-full，这是使用mmlab的全家桶的基础
#1. 直接从github上拉下来，当作代码库，直接进行操作
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip3 install -e .
#2. 使用pip或是mim安装，仅仅把mmcls看作一个工具包
#pip install mmclassification or mim install mmclassification

#optional,我一般习惯把所有依赖都安装上，免得后面需要的时候一个一个安装
#pip install -r requirements.txt
```

## 第二步 构建数据集
下载好数据集，避免不必要的错误，为把中文都改了，修改后文件结构为：
```
data
├── led
│   ├── good
│   ├── bad
│   ├── bad_aug
│   ├── test
│   ├── train.txt
│   ├── val.txt
```
其中train.txt，val.txt通过：
```shell
pyhton tools/datasets_split_led.py
```
生成。

## 第三步 修改模型配置
mmclssification包含众多模型，包括:
 VGG,
 ResNet,
 ResNeXt,
 SE-ResNet,
 SE-ResNeXt,
 RegNet,
 ShuffleNetV1,
 ShuffleNetV2,
 MobileNetV2,
 MobileNetV3,
 Swin-Transformer,
 RepVGG,
 Vision-Transformer,
 Transformer-in-Transformer,
 Res2Net,
 MLP-Mixer,
 DeiT,
 Conformer,
 T2T-ViT,
 Twins,
 EfficientNet,
 ConvNeXt,
 HRNet,
 VAN,
 ConvMixer,
 CSPNet,
 PoolFormer。

 这里，
 ```
configs
├── _base_
│   ├── datasets
│   ├── models
│   ├── schedules
│   ├── default_runtime.py
├──  some model configs
```
我们从mmclassification中的configs下的选取你想要的尝试一个模型配置文件，copy到本代码的config目录下，同时你可以改变schedules目录下的文件，改变训练策略；改变datasets目录的文件，改变数据集；改变当你修改了模型后，你还需要增加或者修改models下的文件，最后修改后不要忘记修改configs目录下的配置文件，当然以上的一切都可以直接修改configs目录下的配置文件(这里比较繁琐，有点绕，主要是要多次尝试)

此外，mmclassifiaction有一些非常有用的工具可以使用：
```shell
1. mmclassification/tools/misc/print_config.py
2. mmclassification/tools/misc/verify_dataset.py
```

## 第四步 模型训练
数据和配置文件写好后就可以直接训练了，这里为直接提供了一个脚本，使用
```
./train.sh 
```
就可以训练了

脚本里面张这样：
```
# 1. base training
CONFIG_FILE=configs/resnet50_1xb32_led.py
RESULT_DIR=training/resnet50_1xb32_led
python tools/train.py ${CONFIG_FILE}  --work-dir ${RESULT_DIR}  --gpu-id  0

# 2. pseudo traning
# CONFIG_FILE=configs/led/swin-tiny_1xb64_led_pseudo.py
# RESULT_DIR=training/swin-tiny_1xb64_led_pseudo
# python tools/train.py ${CONFIG_FILE}  --work-dir ${RESULT_DIR}  --gpu-id  0
```
里面有两步，第一步，最基础的训练，修改CONFIG_FILE为你刚刚配置的模型文件即可，然后再修改一下你的输出文件夹RESULT_DIR

第二步，伪标签学习，这里请大家自行探索哦



## 第五步 生成结果并提交
这里提供了一个脚本
```
./infer_led.sh
```
infer_led.sh具体为：
```
IMAGE_DIR=data/led/test  
CONFIG_FILE="configs/resnet50_1xb32_led.py"  
WEIGHT="training/resnet50_1xb32_led/epoch_50.pth" 
RESULT=resnet50_1xb32_led.csv

python tools/infer_led.py ${IMAGE_DIR}   --config ${CONFIG_FILE[*]}  --checkpoint  ${WEIGHT[*]}   --result_csv_file ${RESULT}\
                # --pu_label  #去掉注释进行伪标签标注，不去则生成结果csv文件
```
调用了tools/infer_led.py 这个python文件，这里还包含一些简单的伪标签学习和模型集成方的代码，这里抛砖引玉供大家参考。


## 第六步 可改进方向
这里为使用resnet50，由于样本量较少，只使用RepeatDataset，没有其他额外调参，训练30epoch时，分数：0.87179。
下面是为认为的可改进方向：
1. 调参；本赛题为： LED灯的色彩和均匀性检测，直观可以看出颜色和形状(比如：完整与否)对模型性能影响较大，mmcls的配置文件更多是正对自然图像（ImageNet）的，在LED灯的色彩和均匀性检测这个任务上，并不算很适配，像随机裁剪、mixup这类在自然图像上非常有用的trick，用在本任务上可能还会带来干扰。
2. 数据增强；训练集良品的样本451个，次品的数目为41个，数据量小，且存在不均衡的现象，数据增强往往能带来不错的增益
3. 伪标签学习；训练集一共才492个，而测试集却有1494个，大量未标注的数据存在，伪标签学习看来也是一个不错的涨点思路
4. 多模型集成，大力出奇迹；单模型无法提升时，多模型集成，往往能出奇制胜

## PS：本人水平有限，如有错误，请大佬们多多包涵，🙏，多多指正
