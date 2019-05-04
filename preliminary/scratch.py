l = ['sky', 'water']
for file in l:
    dir = '/home/gry/sr_place/datasets/OST_train/rain_v2/'
    dir += file
    a = os.listdir(dir)
    a.__len__()
    with open('train.txt', 'a') as f:
        num = 0
        for pic in a:
                path = os.path.join(dir, pic)
                if os.path.getsize(path) > 1000000:
                 	path += "\n"
                 	s = f.write(path)
                 	num += 1
        print(num)
        print("\n")


import os
file = open('train.txt')
while 1:
    line = file.readline()[:-1]
    if not line:
            break
    data.append(line)
import random
slice = random.sample(data, 10)
for pic in slice:
    data.remove(pic)
import readmat
for i in range(20):
    fileset = random.sample(data, 70) + slice
    os.system("rm self_validation_HR/*")
    for file in fileset:
        os.system("cp "+ file +" self_validation_HR/")
    os.system("matlab  -nodesktop -nosplash -nojvm < evaluate_results.m")
    readmat("test_score_"+str(i))


docker run --rm -itd --init \
  --name=train_SRResNet_wgan_18e4D_2 \
  --runtime=nvidia \
  --user="$(id -u):$(id -g)" \
  --volume=$PWD:/app \
  -v /home/nfs:/home/nfs \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  --ipc=host \
  --cpus=20 pytorch-gry /bin/bash -c "cd BasicSR/codes; sh run.sh"

docker run --rm -itd --init \
  --runtime=nvidia \
  --user="$(id -u):$(id -g)" \
  --volume=$PWD:/app \
  -v /home/nfs:/home/nfs \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  --ipc=host \
  --cpuset-cpus="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18" \
  --name=train_SRRes_wgan_D128SN_delayD1e4.json \
  pytorch-gry /bin/bash -c "cd BasicSR/codes; sh run.sh"


animal/cat \
animal/cougar \
animal/fur_1 \
animal/horse \
animal/koala \
animal/raccoon \
animal/bear_1 \
animal/bear_2 \
animal/bird_2 \
animal/bird_1 \
animal/cheetah_2 \
animal/cheetah_1 \
animal/dog_1 \
animal/dog_2 \
animal/dog_0 \
animal/fox_1 \
animal/fox_0 \
animal/lion_2 \
animal/lion_0 \
animal/quokka_1 \
animal/quokka_0 \
animal/tiger_1 \
animal/tiger_0 \
animal/fur_2 \
animal/fur_3 \
animal/fur_0 \
animal/lion_1 \
water/lakesea_2 \
water/lakesea_3 \
water/n07935504 \
water/water_0 \
water/water_1 \
water/water_2 \
water/water_3 \
water/lakesea_0 \
water/lakesea_1 \
building/n03842012 \
building/n02914991 \
building/n03449564 \
 building/n03043693 \
building/brick_1 \
building/brick_2 \
building/brick_3 \
building/brick_4 \
building/brick_5 \
building/brick_6 \
building/n02913152_2 \
building/n02913152_3 \
building/n02913152_4 \
building/n03028079_1 \
building/n03028079_2 \
building/n03028079_3 \
building/n03028079_4 \

building/n04233124_1 \
building/n04233124_2 \
building/n04233124_3 \
building/n04233124_4 \
building/n04233124_5 \

building/building_2 \
building/building_4 \

building/building_6 \
building/building_8 \
building/n02913152_1 \
building/brick_0 \
building/building_3 \
building/building_5 \
building/building_7 \
building/building_0 \
building/building_1 \
mountain/mountain_2 \
mountain/mountain_3 \
mountain/mountain_4 \
mountain/mountain_5 \
mountain/mountain_6 \
mountain/mountain_7 \
mountain/mountain_8 \
mountain/mountain_9 \
mountain/mountain_10 \
mountain/mountain_1 \
grass/grass_2 \
grass/grass_3 \
grass/grass_4 \
grass/grass_0 \
grass/tallgrass_2 \
grass/tallgrass_0 \
grass/lawn_2 \
grass/lawn_3 \
grass/lawn_0 \
grass/tallgrass_3 \
grass/tallgrass_1 \
grass/grass_1 \
grass/lawn_1 \
plant/plant_2 \
plant/plant_3 \
 plant/plant_4 \
plant/plant_5 \
plant/plant_6 \
plant/plant_7 \
plant/plant_8 \
plant/plant_9 \
plant/plant_10 \
plant/plant_1 \
sky/sky_1 \
sky/cloud_2 \
sky/cloud_3 \
sky/sky_10 \
sky/sky_11 \
sky/sky_12 \
sky/sky_13 \
sky/sky_2 \
sky/sky_3 \
sky/sky_4 \
sky/sky_5 \
sky/sky_6 \
sky/sky_7 \
sky/sky_8 \
sky/sky_9 \
sky/cloud_0 \
sky/cloud_1 \
BSD_png \

D128SN_delayD1e4_50000.pth  VANILLA_125000.pth  WGAN_100000.pth  WG_MINC_18e4D_1_100000.pth  WG_MINC_18e4D_2_100000.pth  WG_MINC_18e4D_3_50000.pth \
D128SN_delayD1e4_75000.pth  VANILLA_150000.pth  WGAN_125000.pth  WG_MINC_18e4D_1_125000.pth  WG_MINC_18e4D_2_125000.pth

for pic in data[0]:
     dict = {}
     dict['name'] = str(pic[0][0])
     dict['Ma'] = np.array(pic[1][0]).astype('float')
     dict['Ma_f1'] = np.array(pic[2][0]).astype('float')
     dict['Ma_f2'] = np.array(pic[3][0]).astype('float')
     dict['Ma_f3'] = np.array(pic[4]).astype('float')
     dict['Ma_s1'] = np.array(pic[5][0]).astype('float')
     dict['Ma_s2'] = np.array(pic[6][0]).astype('float')
     dict['Ma_s3'] = np.array(pic[7][0]).astype('float')
     dict['NIQE'] = np.array(pic[8][0]).astype('float')
     dictset.append(dict)


if [ ! -d "/myfolder" ]; then echo "NO"; fi

100000_G.pth  110000_G.pth  125000_G.pth  140000_G.pth  15000_G.pth   165000_G.pth  35000_G.pth  50000_G.pth  60000_G.pth  75000_G.pth  90000_G.pth \
10000_G.pth   115000_G.pth  130000_G.pth  145000_G.pth  155000_G.pth  20000_G.pth   40000_G.pth  5000_G.pth   65000_G.pth  80000_G.pth  95000_G.pth \
105000_G.pth  120000_G.pth  135000_G.pth  150000_G.pth  160000_G.pth  25000_G.pth   45000_G.pth  55000_G.pth  70000_G.pth  85000_G.pth

ASC_WG_MINC_18e4D_2/ASC_5000_G \
ASC_WG_MINC_18e4D_2/ASC_10000_G \
ASC_WG_MINC_18e4D_2/ASC_15000_G \
ASC_WG_MINC_18e4D_2/ASC_20000_G \
ASC_WG_MINC_18e4D_2/ASC_25000_G \
ASC_WG_MINC_18e4D_2/ASC_30000_G \
ASC_WG_MINC_18e4D_2/ASC_35000_G \
ASC_WG_MINC_18e4D_2/ASC_40000_G \
ASC_WG_MINC_18e4D_2/ASC_45000_G \
ASC_WG_MINC_18e4D_2/ASC_50000_G \
ASC_WG_MINC_18e4D_2/ASC_55000_G \
ASC_WG_MINC_18e4D_2/ASC_60000_G \
ASC_WG_MINC_18e4D_2/ASC_65000_G \
ASC_WG_MINC_18e4D_2/ASC_70000_G \
ASC_WG_MINC_18e4D_2/ASC_75000_G \
ASC_WG_MINC_18e4D_2/ASC_80000_G \
ASC_WG_MINC_18e4D_2/ASC_85000_G \
ASC_WG_MINC_18e4D_2/ASC_90000_G \
ASC_WG_MINC_18e4D_2/ASC_95000_G \
ASC_WG_MINC_18e4D_2/ASC_100000_G \
ASC_WG_MINC_18e4D_2/ASC_105000_G \
ASC_WG_MINC_18e4D_2/ASC_110000_G \
ASC_WG_MINC_18e4D_2/ASC_115000_G \
ASC_WG_MINC_18e4D_2/ASC_120000_G \
ASC_WG_MINC_18e4D_2/ASC_125000_G \
ASC_WG_MINC_18e4D_2/ASC_130000_G \
ASC_WG_MINC_18e4D_2/ASC_135000_G \
ASC_WG_MINC_18e4D_2/ASC_140000_G \
ASC_WG_MINC_18e4D_2/ASC_145000_G \
ASC_WG_MINC_18e4D_2/ASC_150000_G \
ASC_WG_MINC_18e4D_2/ASC_160000_G \
ASC_WG_MINC_18e4D_2/ASC_155000_G \
ASC_WG_MINC_18e4D_2/ASC_165000_G

tmux-1.6-3.el6.x86_64

sh test.sh plant/plant_1 plant/plant_6 plant/plant_8 plant/plant_9 sky/sky_13

sh test.sh building/n04233124_1 building/n04233124_2 building/n04233124_3 building/n04233124_4 building/n04233124_5 water/lakesea_1 water/lakesea_2 water/lakesea_3 animal/fox_0

DIV2K_train_HR/00 \
DIV2K_train_HR/01 \
DIV2K_train_HR/02 \
DIV2K_train_HR/03 \
DIV2K_train_HR/04 \
DIV2K_train_HR/05 \
DIV2K_train_HR/06 \
DIV2K_train_HR/07 \
DIV2K_train_HR/08


[[0.0113437365584950, 0.0838195058022106,	0.0113437365584951],
[0.0838195058022110, 0.619347030557177, 0.0838195058022110],
[0.0113437365584951, 0.0838195058022106,	0.0113437365584951]]

[[0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0],
[0,	0,	0.427450980392157,	0.423529411764706,	0.423529411764706,	0.427450980392157,	0.431372549019608],
[0,	0,	0.427450980392157,	0.427450980392157,	0.427450980392157,	0.427450980392157,	0.431372549019608],
[0,	0,	0.431372549019608,	0.431372549019608,	0.431372549019608,	0.431372549019608,	0.431372549019608],
[0,	0,	0.431372549019608,	0.435294117647059,	0.435294117647059,	0.435294117647059,	0.435294117647059],
[0,	0,	0.435294117647059,	0.435294117647059,	0.435294117647059,	0.439215686274510,	0.435294117647059]]

ASC_WG_MINC_18e4D_2/ASC_170000_G \
ASC_WG_MINC_18e4D_2/ASC_175000_G \
ASC_WG_MINC_18e4D_2/ASC_180000_G \
ASC_WG_MINC_18e4D_2/ASC_185000_G \
ASC_WG_MINC_18e4D_2/ASC_190000_G \
ASC_WG_MINC_18e4D_2/ASC_195000_G \
ASC_WG_MINC_18e4D_2/ASC_200000_G \
ASC_WG_MINC_18e4D_2/ASC_205000_G \
ASC_WG_MINC_18e4D_2/ASC_210000_G \
ASC_WG_MINC_18e4D_2/ASC_215000_G \
ASC_WG_MINC_18e4D_2/ASC_220000_G \
ASC_WG_MINC_18e4D_2/ASC_225000_G \
ASC_WG_MINC_18e4D_2/ASC_230000_G \
ASC_WG_MINC_18e4D_2/ASC_235000_G