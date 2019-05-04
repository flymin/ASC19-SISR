### ATTENTION:
- 进行测试时需要同时提供original图像以及生成的图像，两部分的对应文件名应当一致
- 官方给出的matlab测试脚本测试时间较长，处理50张图像的时间大约为25分钟
- 目前ESRGAN的test过程需要显存1G，100张图像可以在10s左右完成
- Pytorch 1.0版本修改了dataloader的设计，内部的部分函数以及功能都有修改，旧的网络代码可能无法适应Pytorch1.0
- score代码中均去掉了上下左右scale像素的region
- ESRGAN中分别空余了HR和ref两个变量，其中HR用来计算netF和pix，ref用来计算netD

### useful instrucion
- 挂载指令
	```bash
	sudo mount -o rw,fmask=0000,dmask=0000 /dev/sdb3 /mnt/sdb3/
	```
- matlab
	```bash
	matlab  -nodesktop -nosplash -nojvm -r evaluate_results #运行测试脚本
	nohup matlab  -nodesktop -nosplash -nojvm < evaluate_results.m > DBPN_region2 2>&1 &	#后台运行
	```
- 改变图像尺寸
	```python
	import os
	src = os.listdir(".")
	dst = os.listdir("../DBPN_3")
	dstfile = dst[:]
	for pic in dstfile:
		pic = "../DBPN_3/" + pic
	import cv2
	sizedict = {}
	for pic in range(dst.__len__()):
		image = cv2.imread(dstfile[pic])
		sizedict[dst[pic]] = image.shape
	for pic in src:
		image = cv2.imread(pic)
		image = cv2.resize(image, (sizedict[pic][1], sizedict[pic][0]), interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(pic, image)
	```
- 文件名转换python指令
	```python
	import os
	pos1 = -10	# should be changed accordingly
	pos2 = -4	# should be changed accordingly
	a = os.listdir(".")
	b = a[:]
	for temp in range(a.__len__()):
		b[temp] = a[temp][0:pos1]+a[temp][pos2:]
	for i in range(a.__len__()):
		dst = os.path.join(".", b[i])
		os.rename(a[i], dst)
	```
- `watch -n sec "INSTRUCTION"`监视指令
- `du -h --max-depth=1`查看当前目录的文件夹大小
- `disown -h $1`后台执行不断开
- conda使用
	```bash
	conda create -n py36 python=3.6 #创建新环境
	conda activate py36 #激活新环境
	```
- 使用官方的降采样方法，注意这份代码不会创建文件夹，所以需要提前建立并输入已经存在的文件夹
	```bash
	python GEN_LR.py INPUTDIR OUTPUTDIR
	```
- 训练ESRGAN
	1. 首先裁剪DIV2K的数据（[作者指出DIV2K图像过大](https://github.com/xinntao/BasicSR/tree/master/codes/data), 并且尝试直接训练会爆显存（单卡），网络并行方式还未摸清）
	2. 根据生成的裁剪后（sub）图像产生LR数据集
	1. 编辑options/train/train_ESRGAN.json
	```json
	//修改数据集路径
	"datasets":"train":"dataroot_HR"&"dataroot_LR"
	//修改验证数据集，设置方式同test参数
	"datasets":"val"
	//修改path，添加pretrain_model
	"path"
	```
	2. 运行指令
	```shell
	python -u train.py -opt options/train/train_ESRGAN.json
	```
- DBPN网络
	```python
	python3    eval_gan.py 		#Testing GAN for PIRM2018
	python3    eval.py 			#Testing
	```
- DSRN网络
	1. 注意requirements.txt中的库依赖(tensorflow-gpu只能在python3.5.x中使用)
- SSH反向代理与端口转发[利用ssh反向代理以及autossh实现从外网连接内网服务器](https://www.cnblogs.com/kwongtai/p/6903420.html)
	> 反向代理
	> `ssh -fCNR`
	>
	> 正向代理
	> `ssh -fCNL`
	```
	-f 后台执行ssh指令
	-C 允许压缩数据
	-N 不执行远程指令
	-R 将远程主机(服务器)的某个端口转发到本地端指定机器的指定端口
	-L 将本地机(客户机)的某个端口转发到远端指定机器的指定端口
	-p 指定远程主机的端口
	```

	```bash
	# On machine A with LAN network
	ssh -fCNR [B机器IP或省略]:[B机器端口]:[A机器的IP]:[A机器端口] [登陆B机器的用户名@服务器IP]
	ssh -fCNR 7280:localhost:22 root@123.123.123.123
	ps aux | grep ssh
	
	# On machine B with WAN network
	ssh -fCNL [A机器IP或省略]:[A机器端口]:[B机器的IP]:[B机器端口] [登陆B机器的用户名@B机器的IP]
	ssh -fCNL *:1234:localhost:7280 localhost
	
	# autossh, 首先保证A机器可以使用rsa登录到B机器
	autossh -M 7281 -fCNR 7280:localhost:22 root@123.123.123.123
	# 7280用来建立ssh反向链接，7281用来作为保持链接隧道
	```

### got so far
- 超分辨率竞赛[已知PIRM和NTIRE(CVPR related)]的使用的降采样方法未必一样，ASC给出的降采样采用的双三次插值，这一点与ESRGAN使用的训练数据采用的方法一致。
- GAN网络生成的图像可以让人眼感受到很逼真的感觉，但细节部分的重建未必和原图一致（差异的产生来自于GAN的感知性，这也是影响GAN不同类别样本迁移的因素）；而非GAN方法的超分辨率解决方案立足于生成更贴近原图的高分辨率图像
- ESRGAN给出的训练方法中并没有支出使用那个数据集进行的训练以及具体的训练步骤，但强调了通常使用DIV2K数据集，**准备在训练中使用[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集**
- 找到NTIRE2017 1st的代码（pytorch），这个代码是在pytorch 1.0版本制作的，1.0版本中修改了dataloader的设计，需要使用pytorch1.0，安装之后可执行。**待细读[知乎专栏：NTIRE2017方法总结](https://zhuanlan.zhihu.com/p/39930043)**
	
	>注意安装pytorch需要切换镜像源

	```shell
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
	
	# for legacy win-64
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
	conda config --set show_channel_urls yes
	conda install pytorch torchvision #需要删除-c指定源的选项，否则镜像不起作用
	```

- 关于多机多卡的并行训练，示例代码[如何使用分布式训练（多机多卡）](https://blog.genkun.me/post/pytorch-faq/#%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1)
	
	```python
	import torch
	from torch.nn.parallel import DistributedDataParallel
	import torch.distributed as dist
	 
	# 每个节点都要用以下语句初始化
	# init_method 是 master 机器的 IP 和端口，worker 们只需要与 master 机器通信
	# world_size 是结点数量
	# rank 是该 worker 的序号，不同结点的 rank 是不同的
	dist.init_process_group(backend='gloo', init_method=tcp://192.168.1.101:2222,
	    world_size=5, rank=0)
	 
	# 定义 dataset
	trainset = Market1501(root=args.dataset, data_type='train', transform=transform)
	# DistributedSampler 的作用是将训练数据随机采样，送给不同的结点去 forward 和 backward
	train_sampler = DistributedSampler(trainset)
	# 定义 dataloader
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
	    shuffle=(train_sampler is None), num_workers=5, pin_memory=True, sampler=train_sampler)
	 
	# 定义分布式的模型
	net = Net()
	net = net.cuda()
	net = DistributedDataParallel(net)
	 
	for epoch in range(20):
	 
	    # 设置当前的 epoch，为了让不同的结点之间保持同步。
	    train_sampler.set_epoch(epoch)
	 
	    # 以下的过程就是跟平常非分布式的一样了
	    for i, data in enumerate(dataloder):
	        x, label = data
	        x, label = Variable(x).cuda(), Variable(label).cuda()
	 
	        output = net.forward(x)
	        loss = criterion(output, label)
	 
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	```

	>需要每个节点启动一个独立的Python训练脚本（脚本都是相同的）
	>
	>更多参考[分布式系统：在多台机器上训练模型](https://www.jqr.com/article/000537#h4--)，包括启动方式的参考

### Messages & Results

**这里记录的是最原始的资料收集记录，可能与仓库真实情况有所出入**

#### 测试记录概要

- [x] test for DBPN，数据已产生，matlab score
- [x] [LapSRN](https://github.com/twtygqyy/pytorch-LapSRN)，官方认可的第三方实现，给出了4x的pretrain model，但采用了matlab矩阵方式读取，可能需要修改dataloader
	> 代码只能产生灰度图像，根据readme显示gan的test作者还没有写完
- [x] [SRDenseNet](https://github.com/twtygqyy/pytorch-SRDenseNet) ，只有使用灰度图像训练和测试的版本，无法适用
- [x] [SRResNet](https://github.com/twtygqyy/pytorch-SRResNet) ，只有使用灰度图像训练和测试的版本，无法适用
	> 以上三份代码来自于同一套实现框架，因此pytorch版本中都没有给出三通道的处理
- [x] [DRRN1](https://github.com/jt827859032/DRRN-pytorch) ，有预训练模型（291张图）
	> 似乎是和LapSRN使用了相同的baseline，没有提供图像输出
- [x]	[DRRN2](https://github.com/yiyang7/cs231n_proj)，有CNNbase的预训练和GANbase的预训练两种
	> 2019.1.25仓库有更新，集成了多种模型，但只提供了人脸数据集的训练结果
- [x] [RCAN](https://github.com/yiyang7/cs231n_proj) ，有预训练模型，据说已经被merge in EDSR，不是GAN方式
- [x] DSRN只有tensorflow的模型，预训练模型的加载未找到，需要lingvo，搁置
- [x] [IDN](https://github.com/lizhengwei1992/IDN-pytorch) ，只有第三方实现的pytorch（初步预测和原文有些差距，只有train没有test），原文采用了caffe+TensorFlow两种实现方式
- [x] [ZSSR](https://github.com/assafshocher/ZSSR) ，只有tensorflow的x2模型
- [x] [SFTGAN](https://github.com/xinntao/SFTGAN) ，有完整的模型和测试，但这个仓库提供的并不是超分辨率的复原方法，而是在原有的超分辨率图像基础之上做出的真实性纹理复原
	> 经测试发现通过SFTGAN之后图像会比4x尺寸有所减小，所以无法使用ASC脚本进行测试
- [x] [ESRGAN](https://github.com/xinntao/ESRGAN) ，怀疑之前的模型加载有问题的，重新确认之后数据无误
- [x] [SRGAN With WGAN](https://github.com/JustinhoCHN/SRGAN_Wasserstein) ，[知乎专栏-SRGAN With WGAN，让超分辨率算法训练更稳定](https://zhuanlan.zhihu.com/p/37009085)
- [x] [DFFNet](https://github.com/XuwangNUAA/DFFNet) ，网上目前只有测试代码，从论文结果来看是目前PSNR指标做的最好的
- [x] [ESRGAN-PIRM](https://github.com/xinntao/ESRGAN/blob/master/QA.md) ，作者发现了PI只并不能达到很好地反应视觉评价的效果，因此做了一些额外的调整来优化PI值，这里给出模型和后期优化
- [x] [MBSR](https://github.com/pnavarre/pirm-sr-2018) ，PIRM region3中第二名，没有训练代码，但同时提供了三个region的权重文件
- [x] [EPSR](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw) ，PIRM region3中的第三名，从命名来看有训练代码，同时提供了三个region的权重文件
	> 需要使用pytorch 1.0以下的版本运行
- [x] [PESR](https://github.com/thangvubk/PESR) ，PIRM region3第四名，没有已发表的论文，引入了perceptual weight，可以整合两种评价指标产生的结果

**最终选择了ESRGAN作为baseline model，并整合了其他方发进行改进**

#### 测试结果

_**更准确详细的数据见Excel**_

* set5: 使用官方的数据集生成的高分辨率图像，基于SRGAN x4 model, trained on DIV2K, w/o BN, bicubic downsampling.(由于缺少HR图像无法给出测试结果)
* set_my: 使用PIRM2018val数据集，100张图片，采用ASC给出的方式进行降采样，基于SRGAN x4 model, trained on DIV2K, w/o BN, bicubic downsampling的训练结果给出的测试结果
	- ASC测试结果(SRGAN)：
	```
		Your perceptual score is: 2.214
		Your RMSE is: 15.4226
	```
	- ASC测试结果(ESRGAN)：
	```
		Your perceptual score is: 2.5454
		Your RMSE is: 16.4194
	```
	> **看起来很奇怪，根据题目描述方法全部一样，那么为什么ESRGAN会比SRGAN更差？**
	- ASC测试结果([EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)):
	```
		Your perceptual score is: 5.2387
		Your RMSE is: 11.1636
	```
	- 直接使用高分辨率图像运行测试程序
	```
		Your perceptual score is: 2.2818
		Your RMSE is: 0
	```
	> **这就更奇怪了，PI参数的计算问题？**
	- ASC测试结果([DBPN](https://github.com/alterzero/DBPN-Pytorch)):
	```
		Namespace(chop_forward=False, gpu_mode=True, gpus=1, input_dir='Input', model='models/PIRM2018_region2.pth', model_type='DBPNLL', output='Results/', seed=123, self_ensemble=False, testBatchSize=1, test_dataset='PIRM_Self-Val_set', threads=1, upscale_factor=4)
		Your perceptual score is: 2.2923
		Your RMSE is: 12.5282
		Region3
		Your perceptual score is: 2.1273
		Your RMSE is: 13.245
	```
	- ASC测试结果([SRGAN With WGAN](https://github.com/JustinhoCHN/SRGAN_Wasserstein))
	```
		Your perceptual score is: 2.2843
		Your RMSE is: 15.0629
	```
	ASC测试结果([DFFNet](https://github.com/XuwangNUAA/DFFNet))
	```
		Your perceptual score is: 5.0693>> 
		Your RMSE is: 11.1077
	```
* DIV2K数据集，训练集800张图像，100张val，100张test

	> selfgen中是使用ASC的down-sampl生成的训练集图像<br>
	> 带有sub标签的是使用ESRGAN代码产生的裁剪图像(train集32208张，val集4144张)
