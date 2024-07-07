# 不可感知性强的人脸属性编辑主动防御

![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)

![PyTorch 1.13.1](https://img.shields.io/badge/pytorch-1.13.1-green.svg?style=plastic)


> **摘要:** *为消除恶意人脸属性编辑技术带来的信息安全隐患,主动防御技术应运而生.现有方法一部分基于梯度攻击生成对抗扰动,需多次迭代,速度较慢,而另一部分基于GAN生成对抗扰动,速度较快.但现有这类方法仍有所不足,其一是难以兼顾防御效果与生成扰动的不可感知性,其二是未着重破坏人脸图像主要伪造区域.针对上述问题,本文基于GAN提出了一种不可感知性更强的人脸属性编辑主动防御方法.该方法通过在解码器中引入所提高频信息补偿并在编码器中引入所提多级密集连接,使生成器生成的对抗扰动能在保证防御效果的前提下有更强的不可感知性.该方法在训练中引入所提人脸显著性对抗损失,使扰动能更好地破坏人脸伪造区域.本文在单个模型和跨模型防御场景下进行了实验,结果表明,所提方法相比现有主动防御方法能生成不可感知性更强的对抗扰动,且对目标模型均能取得较高的防御成功率.*


## 数据集与预训练模型

数据集使用CelebA数据集,具体下载可参考[CMUA-Watermark](https://github.com/VDIGPKU/CMUA-Watermark). 

人脸属性编辑模型StarGAN、AttentionGAN、AttGAN、FGAN的预训练权重与本文训练的U2Net的预训练权重及自建数据集均可从百度网盘进行下载,并相应修改settings.json中的路径配置微数据集解压后的具体存储路径.

点击([Pretrained Weights](https://pan.baidu.com/s/1AMhVnrcB4OIUOwym8tP7eA?pwd=msch))下载本文方法涉及的预训练权重,按照路径存放相应的权重.

```xml
AttentionGAN路径：'./AttentionGAN/AttentionGAN_v1_multi/checkpoints/celeba_256_pretrained/'
AttGAN路径：'./AttGAN/output/256_shortcut1_inject0_none/checkpoint/'
FGAN路径：'./fgan/ckpts/'
StarGAN路径：'./stargan/stargan_celeba_256/models/'
U2Net路径：'./face_sod_u2net/ckpts/'
```

点击([Face-SOD-DataSet](https://pan.baidu.com/s/1f-SRP5J-9OEs4TFNT4_10w?pwd=r33j))下载本文自建人脸显著性数据集,包含了12000张训练集图片与3000张测试集图片,可用于训练各种显著性检测模型用于检测人脸显著性区域任务.


## 训练
执行下列命令以训练模型对AttGAN进行攻击

```python
  python -m train_sfgan.att_sod_loss
```
执行下列命令以训练模型对AttentionGAN进行攻击

```python
  python -m train_sfgan.attention_sod_loss
```
执行下列命令以训练模型对FGAN进行攻击

```python
  python -m train_sfgan.fgan_sod_loss
```
执行下列命令以训练模型进行集成攻击

```python
  python -m train_sfgan.ensemble
```

## 相关工作
本工作部分基于 [CMUA-Watermark](https://github.com/VDIGPKU/CMUA-Watermark) 、 [WaveGAN](https://github.com/kobeshegu/ECCV2022_WaveGAN) 和 [DenseNet](https://github.com/bamos/densenet.pytorch).感谢他们高质量的工作.
