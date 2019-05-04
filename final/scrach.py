'''
matplot绘图，subplot
注意cv2通道顺序问题
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
'''
import matplotlib.pylab as plt
plt.figure("contrast")
plt.suptitle("contrast")
plt.subplot(2,2,1), plt.title("origin1")
plt.imshow(org1), plt.axis("off")
plt.subplot(2,2,2), plt.title("origin2")
plt.imshow(org2), plt.axis("off")
plt.subplot(2,2,3), plt.title("new1")
plt.imshow(img1), plt.axis("off")
plt.subplot(2,2,4), plt.title("new2")
plt.imshow(img2), plt.axis("off")
# plt.imshow() 	# 想要保存图像必须在save之前调用show
plt.savefig("contrast.png")
plt.close()		# 关闭当前图像
# plt.close("all") # 关闭所有图像

import os
import cv2
inputDir = "lfw"
outputDir = "lfw_crop"
pics = os.listdir(inputDir)
for pic in pics:
	img = cv2.imread(os.path.join(inputDir, pic))
	img = img[1:-1, 1:-1, :]
	status = cv2.imwrite(os.path.join(outputDir, pic), img)

# 仿射矩阵转化，这个函数不正确
img = torch.from_numpy(src_img.transpose(2,0,1))
img = img.unsqueeze(0)
h = 216
w = 176
theta = param
theta[0,0] = param[0,0]
theta[0,1] = param[0,1]*h/w
theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
theta[1,0] = param[1,0]*w/h
theta[1,1] = param[1,1]
theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
theta = torch.from_numpy(theta).unsqueeze(0)
temp = torch.nn.functional.affine_grid(theta, torch.Size((1,3,112,96)))
test = torch.nn.functional.grid_sample(img.double(), temp)


for line in lines:
    l = line.replace('\n','').split(' ')
    filename = l[0]
    name = l[1]
    if name not in dic.keys():
    	dic[name] = []
    dic[name].append(filename)

def SaltAndPepper(src,percetage):  
    SP_NoiseImg=src 
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1]) 
    for i in range(SP_NoiseNum): 
        randX=random.random_integers(0,src.shape[0]-1) 
        randY=random.random_integers(0,src.shape[1]-1) 
        if random.random_integers(0,1)==0: 
            SP_NoiseImg[randX,randY]=0 
        else: 
            SP_NoiseImg[randX,randY]=255 
    return SP_NoiseImg

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

plt.figure("contrast")
plt.suptitle("contrast")
img = cv2.cvtColor(cv2.imread("Aaron_Guiel_0001_HR.png"), cv2.COLOR_BGR2RGB)
plt.subplot(3,3,1), plt.title("HR")
plt.imshow(img), plt.axis("off")
img = cv2.cvtColor(cv2.imread("Aaron_Guiel_0001_LR.png"), cv2.COLOR_BGR2RGB)
plt.subplot(3,3,2), plt.title("LR")
plt.imshow(img), plt.axis("off")
img = cv2.cvtColor(cv2.imread("Aaron_Guiel_0001_LR_motion.png"), cv2.COLOR_BGR2RGB)
plt.subplot(3,3,3), plt.title("LR_motion")
plt.imshow(img), plt.axis("off")
img = cv2.cvtColor(cv2.imread("Aaron_Guiel_0001_LRcuSR.jpg"), cv2.COLOR_BGR2RGB)
plt.subplot(3,3,4), plt.title("LRcuSR-0.5266")
plt.imshow(img), plt.axis("off")
img = cv2.cvtColor(cv2.imread("Aaron_Guiel_0001_SR.png"), cv2.COLOR_BGR2RGB)
plt.subplot(3,3,5), plt.title("SR-0.9294")
plt.imshow(img), plt.axis("off")
img = cv2.cvtColor(cv2.imread("Aaron_Guiel_0001_SR_motion.png"), cv2.COLOR_BGR2RGB)
plt.subplot(3,3,6), plt.title("SR_motion-0.9322")
plt.imshow(img), plt.axis("off")
img = cv2.cvtColor(cv2.imread("Aaron_Guiel_0001_SR_motion_flip.png"), cv2.COLOR_BGR2RGB)
plt.subplot(3,3,9), plt.title("SR_motion_flip-0.9370")
plt.imshow(img), plt.axis("off")
# plt.imshow() 	# 想要保存图像必须在save之前调用show
plt.savefig("contrast.png")
plt.close()		# 关闭当前图像


import os
import cv2
dir = os.listdir(".")
paths = []
for d in dir:
	for pic in os.listdir(d):
		path = os.path.join(d, pic)
		img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
		if img.ndim == 2:
			paths.append(path)

num = []
for key in dic.keys():
    num.append(dic[key].__len__())

with open("../CASIA_test.txt", 'w') as f:                                                                                                                                                                                               
    for key in test:                                                                                                                                                                                                                    
        for fi in dic[key]:                                                                                                                                                                                                         
                status = f.write(fi + "\n")
with open("../CASIA_train.txt", 'w') as f:
    for key in train:
        for fi in dic[key]:
            status = f.write(fi + "\n")

with open("../CASIA_train_2.txt", 'w') as f:
	for key in train:
		sel = random.sample(dic[key], 2)
		for fi in sel:
			status = f.write(fi + "\n")
with open("../CASIA_test_2.txt", 'w') as f:
	for key in test:
		sel = random.sample(dic[key], 2)
		for fi in sel:
			status = f.write(fi + "\n")

import re


def getNumber(string):
    pnum = re.compile(r"\d+(,\d+)*")
    enum = re.compile(r"\d+\.\d+(e[+\-]\d+)?")
    match = enum.match(string)
    if match is None:
        match = pnum.match(string)
        if match is None:
            return None
        else:
            match = int(match.group().replace(",", ""))
    else:
        match = float(match.group())
    return match


def convert(filename, items, vals):
    val_p = ''
    for val in vals:
        val_p += val + "|"
    if val_p.endswith("|"):
        val_p = val_p[:-1]
        val_p = re.compile(val_p)
        lines = []
        space = re.compile(r'\s+')
        start = False
    with open(filename) as f:
        for line in f:
            if start:
                lines.append(line)
                continue
            if line.count("Start training"):
                start = True
    dic = {}
    it = 0
    for item in items:
        dic[item] = []
    dic["val_CASIA"] = []
    dic["val_lfw"] = []
    dic["val_celeb"] = []
    dic["val_iter"] = []
    for line in lines:
        line = re.sub(space, "", line)
        for item in items:
            pos = line.find(item)
            if pos == -1:
                continue
            num = getNumber(line[pos + len(item) + 1:])
            assert num is not None, \
                "line: {} with error for {}".format(line, item)
            dic[item].append(num)
            if item == 'iter':
                it = num
        if val_p.search(line):
            dic["val_iter"].append(it)
        for item in vals:
            pos = line.find(item)
            if pos != -1:
                num = getNumber(line[pos + len(item):])
                assert num is not None, \
                    "line: {} with error for {}".format(line, item)
                dic["val_" + item].append(num)
            else:
                continue
    return dic


if __name__ == '__main__':
    a = convert('train_190411-084446.log', ['iter', 'l_g_total'], ['celeb', 'lfw', 'CASIA'])


with open("CASIA-WebFace_11296_BC_cos.csv") as f:
    lines = [line.rstrip("\n") for line in f]
lines.__len__()
pairs = []
for line in lines:
    name, cos = line.split(",")[1:]
    pairs.append((cos, name))

pairs.sort()

N = pairs.__len__()
N = int(N * 0.4)

with open("FFHR_cascade_target.txt", 'w') as f:
    num = 0
    for pair in pairs:
        if num < N:
            s = f.write(pair[1] + "\t1\n")
        else:
            s = f.write(pair[1] + "\t0\n")
        num += 1

whole = "/home/share/FFHR_whole.txt"
sub = "/home/share/divide/train_child.txt"
with_dir = "/home/share/divide/train_child_dir.txt"
import os
with open(whole) as f:
    whole = [line.rstrip("\n") for line in f]

with open(sub) as f:
    sub = [line.rstrip("\n") for line in f]

new = []
for pic in whole:
    name = os.path.split(pic)[1]
    if name in sub:
        new.append(pic)

with open(with_dir, "w") as f:
    for line in new:
        status = f.write(line + "\n")

mkdir \
child_ChConvert \
child_flip  \
child_ill  \
child_ill_d

import os
import cv2
outputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere/child_flip"
inputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere"
filename = "train_child_dir.txt"
with open(filename) as f:
    lines = [line.rstrip("\n") for line in f]

for pic in lines:
    path = os.path.join(inputDir, pic)
    img = cv2.imread(path)
    h_flip = cv2.flip(img, 1)
    path = os.path.join(outputDir, os.path.split(pic)[1])
    assert cv2.imwrite(path, h_flip)

import os
import cv2
outputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere/child_ChConvert"
inputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere"
filename = "train_child_dir.txt"
with open(filename) as f:
    lines = [line.rstrip("\n") for line in f]

for pic in lines:
    path = os.path.join(inputDir, pic)
    img = cv2.imread(path)
    cConvert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path = os.path.join(outputDir, os.path.split(pic)[1])
    assert cv2.imwrite(path, cConvert)

import os
import cv2
outputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere/child_ill"
inputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere"
filename = "train_child_dir.txt"
with open(filename) as f:
    lines = [line.rstrip("\n") for line in f]

alpha = 0.2
for pic in lines:
    path = os.path.join(inputDir, pic)
    img = cv2.imread(path)
    ill = img * (1+alpha)
    path = os.path.join(outputDir, os.path.split(pic)[1])
    assert cv2.imwrite(path, ill)

import os
import cv2
outputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere/child_ill_d"
inputDir = "/home/gry/Training_data_for_FaceSR/Flickr_HR_sphere"
filename = "train_child_dir.txt"
with open(filename) as f:
    lines = [line.rstrip("\n") for line in f]

alpha = 0.2
for pic in lines:
    path = os.path.join(inputDir, pic)
    img = cv2.imread(path)
    ill = img * (1-alpha)
    path = os.path.join(outputDir, os.path.split(pic)[1])
    assert cv2.imwrite(path, ill)