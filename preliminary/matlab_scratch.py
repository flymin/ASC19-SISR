def rgb2ycbcr(img, only_y=True):# 这个函数在ESRGAN代码中实现了，util.py
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def convert_shave_image(img, shave_width):
    image_ychannel = rgb2ycbcr(img)
    shaved = image_ychannel[shave_width:-shave_width, shave_width:-shave_width]
    shaved = shaved.astype("double")
    return shaved

import numpy as np
import cv2
from scipy.special import gamma
from scipy.fftpack import fft, fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import interp1d
import torch
import math
img = cv2.imread("1_G.png")   # 使用cv2读取进来的是bgr通道
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

shaved = convert_shave_image(img, 4)

h = torch.tensor(
    [[0.0113437365584950, 0.0838195058022106, 0.0113437365584951],
     [0.0838195058022110, 0.619347030557177, 0.0838195058022110],
     [0.0113437365584951, 0.0838195058022106, 0.0113437365584951]]).double()
h = h.view(1,1,h.shape[0],h.shape[1])
im1 = torch.tensor(shaved).double()
im1 = im1/255.

im_f = torch.nn.functional.conv2d(im1.view(1,1,im1.shape[0],im1.shape[1]), h, padding=1)
im_f = im_f.squeeze()
im2 = im_f[1::2, 1::2]

im_f = torch.nn.functional.conv2d(im2.view(1,1,im2.shape[0],im2.shape[1]), h, padding=1)
im_f = im_f.squeeze()
im3 = im_f[1::2, 1::2]

def dct(arg1):
    arg1 = np.array(arg1, dtype="float")
    n, m = arg1.shape
    y = np.zeros((2*n, m), "float")
    y[:n, :] = arg1
    y[n:, :] = arg1[::-1]
    yy = fft(y.T).T
    # ww = (exp(-i*(0:n-1)*pi/(2*n))/sqrt(2*n)).';
    ww = (np.exp(-1.0j*np.array([range(n)])*np.pi/(2*n))/np.sqrt(2*n)).T
    # ww(1) = ww(1) / sqrt(2);
    ww[0] = ww[0] / np.sqrt(2)
    # b = ww(:,ones(1,m)).*yy(1:n,:);
    b = np.tile(ww, (1,m))*yy[:n, :]
    if np.isreal(arg1).all(): b = np.real(b)
    return b

def dct2(arg1):
    b = dct(dct(arg1).T).T
    return b

def gama_gen_gauss(I):
    mean_gauss = np.mean(I)
    var_gauss = np.var(I, ddof=1)
    mean_abs=np.mean(np.abs(I-mean_gauss))**2
    rho=var_gauss/(mean_abs+0.0000001)
    # g=0.03:0.001:10;
    g = np.arange(0.03,10.001,0.001)
    r=gamma(1./g)*gamma(3./g)/(gamma(2./g)**2)
    idx=np.argmin(np.abs(r-rho))       # attention start from 0, not from 1 as matlab
    return g[idx]

def gama_dct(I):
    temp1 = dct2(I)
    temp2 = temp1.T.reshape([-1,1])
    temp3 = temp2[1:,:]
    return gama_gen_gauss(temp3)

def blkproc(I, func, block=3, padding=2):
    padding_dict = {0:2, 1:4, 2:3}
    (m,n) = I.shape
    m_p = m + padding + padding_dict[m%block]
    n_p = n + padding + padding_dict[n%block]
    I_p = np.zeros([m_p, n_p])
    I_p[padding:m+padding, padding:n+padding] = I
    size = block + 2*padding
    result = np.zeros([math.ceil(m/block), math.ceil(n/block)])
    m_range = np.arange(0, m, block)
    n_range = np.arange(0, n, block)
    for i in range(m_range.__len__()):
        for j in range(n_range.__len__()):
            B = I_p[m_range[i]:m_range[i]+size, n_range[j]:n_range[j]+size]
            result[i,j] = func(B)
    return result

def coeff_var_gen_gauss(I):
    std_gauss=np.std(np.abs(I.reshape([-1])), ddof=1)
    mean_abs=np.mean(np.abs(I.reshape([-1])))
    rho=std_gauss/(mean_abs+0.0000001)
    return rho

def coeff_var_dct(I):
    temp1 = dct2(I)
    temp2 = temp1.T.reshape([-1,1])
    temp3 = temp2[1:,:]
    return coeff_var_gen_gauss(temp3)

def oriented1_dct_rho_config3(I):
    temp=dct2(I)
    nn = I.shape[0]
    eps=0.00000001
    temp1=[]
    if nn==5 :
        temp1=np.concatenate((temp[0,1:], temp[1,2:], temp[2,-1]));
    elif nn==7 :
        temp1=np.concatenate((temp[0,1:], temp[1,2:], temp[2,4:], temp[3,5:]))
    elif nn==9 :
        temp1=np.concatenate((temp[0,1:], temp[1,2:], temp[2,4:], \
                              temp[3,5:], temp[4,7:], temp[5,8:]))
    std_gauss=np.std(np.abs(temp1[:]), ddof=1)
    mean_abs=np.mean(np.abs(temp1[:]))
    g1=std_gauss/(mean_abs+eps)
    return g1

def oriented2_dct_rho_config3(I):
    temp=dct2(I)
    nn = I.shape[0]
    eps=0.00000001
    temp2=[]
    if nn==5 :
        temp2=np.concatenate((temp[1,1:2], temp[2,2:4], temp[3,2:], temp[4, 3:]));
    elif nn==7 :
        temp2=np.concatenate((temp[1,1:2], temp[2,2:4], temp[3,2:5], \
                              temp[4,3:], temp[5,4:], temp[6,4:]))
    elif nn==9 :
        temp2=np.concatenate((temp[1,1:2], temp[2,2:4], temp[4,3:7], \
                              temp[5,4:8], temp[6,4:], temp[7,5:], temp[8,6:]))
    std_gauss=np.std(np.abs(temp2[:]), ddof=1)
    mean_abs=np.mean(np.abs(temp2[:]))
    g2=std_gauss/(mean_abs+eps)
    return g2

def oriented3_dct_rho_config3(I):
    temp=dct2(I)
    nn = I.shape[0]
    eps=0.00000001
    temp2=[]
    if nn==5 :
        temp3=np.concatenate((temp[1:,0], temp[2:,1], temp[-1,3:4]));
    elif nn==7 :
        temp3=np.concatenate((temp[1:,0], temp[2:,1], temp[4:,2], temp[5:,3]))
    elif nn==9 :
        temp3=np.concatenate((temp[1:,0], temp[2:,1], temp[4:,2], \
                              temp[5:,3], temp[7:,4], temp[8:,5]))
    std_gauss=np.std(np.abs(temp3[:]), ddof=1)
    mean_abs=np.mean(np.abs(temp3[:]))
    g3=std_gauss/(mean_abs+eps)
    return g3

def block_dct(im):
    gama_L1 = blkproc(im, gama_dct)
    gama_sorted_temp = np.sort(gama_L1.reshape([-1]))
    gama_count = len(gama_sorted_temp)
    p10_gama_L1=np.mean(gama_sorted_temp[1:math.ceil(gama_count*0.1)])
    p100_gama_L1=np.mean(gama_sorted_temp)

    feature = [p10_gama_L1, p100_gama_L1]

    coeff_var_L1 = blkproc(shaved, coeff_var_dct)
    cv_sorted_temp = np.sort(coeff_var_L1.reshape([-1]))
    cv_count = len(cv_sorted_temp)
    p10_last_cv_L1=np.mean(cv_sorted_temp[math.floor(cv_count*0.9)-1:])
    p100_cv_L1=np.mean(cv_sorted_temp[:])

    feature = feature + [p10_last_cv_L1, p100_cv_L1]

    ori1_rho_L1 = blkproc(im, oriented1_dct_rho_config3)
    ori2_rho_L1 = blkproc(im, oriented2_dct_rho_config3)
    ori3_rho_L1 = blkproc(im, oriented3_dct_rho_config3)
    temp_size=ori1_rho_L1.shape
    var_temp=np.zeros(temp_size)

    for i in range(temp_size[0]):
        for j in range(temp_size[1]):
            var_temp[i,j]=np.var([ori1_rho_L1[i,j], ori2_rho_L1[i,j], ori3_rho_L1[i,j]], ddof=1)
    ori_rho_L1=var_temp

    ori_sorted_temp = np.sort(ori_rho_L1.reshape([-1]))
    ori_count = len(ori_sorted_temp)
    p10_last_orientation_L1=np.mean(ori_sorted_temp[math.floor(ori_count*0.9)-1:])
    p100_orientation_L1=np.mean(ori_sorted_temp[:])

    feature = feature + [p10_last_orientation_L1, p100_orientation_L1]
    return feature

#====================================================================#
#       feature one ends here                                        #
#====================================================================#

_no_value = object()
def steer2HarmMtx(harmonics, angles=_no_value, evenodd='even'):
    harmonics = harmonics.reshape(1,-1)
    numh = 2*harmonics.shape[1] - (harmonics == 0).any()
    if angles is _no_value:
        angles = np.pi * np.arange(numh).reshape([-1,1])/numh
    else:
        angles = angles.reshape([-1,1])
    imtx = np.zeros([angles.shape[0],numh])
    col = 0
    for h in harmonics.reshape(-1):
        args = h * angles
        if h == 0:
            imtx[:,col] = np.ones(angles.shape)
            col = col+1
        elif evenodd == 'even':   # evenodd = 0
            imtx[:,col] = np.cos(args[:,0])
            imtx[:,col+1] = np.sin(args[:,0])
            col = col+2
        elif evenodd == 'odd':  # evenodd = 1
            imtx[:,col] = np.sin(args[:,0])
            imtx[:,col+1] = -np.cos(args[:,0])
            col = col+2
        else:
            print('EVEN_OR_ODD should be the string  EVEN or ODD')
    r = np.linalg.matrix_rank(imtx)
    if r != numh and r != angles.shape[0]:
        print('WARNING: matrix is not full rank')
    mtx = np.linalg.pinv(imtx)
    return mtx

def rcosFn(width = 1, position = 0, values = None):
    if values is None:
        values = [0,1]
    sz = 256
    X = np.pi*np.arange(-sz-1,2).reshape([1,-1])/(2*sz)
    Y = values[0] + (values[1]-values[0]) * np.cos(X)**2
    Y[0,0] = Y[0,1]
    Y[0,sz+2] = Y[0,sz+1]
    X = position + (2*width/np.pi) * (X + np.pi/4)
    return X, Y

def pointOp(im, lut, origin, increment):
    # 这个函数有对应matlab接口的c实现
    X = origin + increment*np.arange(lut.shape[0]*lut.shape[1])
    Y = lut.reshape(-1)
    res = interp1d(X,Y, kind='linear', fill_value='extrapolate')\
            (im.reshape(-1)).reshape(im.shape)
    return res

def factorial(num):
    if isinstance(num,int):
        return math.factorial(num)
    res = np.ones(num.shape)
    ind = np.where(num > 0)
    if ind[0].__len__() > 0:
        subNum = num[tuple(ind)]
        res[tuple(ind)] = subNum * factorial(subNum-1)
    return res

def buildSFpyrLevs(lodft,log_rad,Xrcos,Yrcos,angle,ht,nbands):
    if (ht <= 0):
        lo0 = ifft2(ifftshift(lodft))
        pyr = np.real(lo0.reshape(-1))
        pind = np.array([lo0.shape])
        return pyr, pind
    bands = np.zeros((np.prod(lodft.shape), nbands))
    bind = np.zeros((nbands,2))
    Xrcos = Xrcos - np.log2(2)
    lutsize = 1024
    Xcosn = np.pi*np.arange(-(2*lutsize+1),(lutsize+2)).reshape([1,-1])/lutsize  # [-2*pi:pi]
    order = nbands-1
    const = (2**(2*order))*(factorial(order)**2)/(nbands*factorial(2*order))
    Ycosn = np.sqrt(const) * (np.cos(Xcosn))**order
    himask = pointOp(log_rad, Yrcos, Xrcos[0,0], Xrcos[0,1]-Xrcos[0,0])
    for b in range(nbands):
        anglemask = pointOp(angle, Ycosn, Xcosn[0,0]+np.pi*b/nbands, Xcosn[0,1]-Xcosn[0,0])
        banddft = ((-1.j)**order) * lodft * anglemask * himask
        band = ifft2(ifftshift(banddft))
        bands[:,b] = np.real(band.reshape(-1))
        bind[b,:]  = band.shape
    dims = np.array(lodft.shape)
    ctr = np.ceil((dims+0.5)/2)
    lodims = np.ceil((dims-0.5)/2)
    loctr = np.ceil((lodims+0.5)/2)
    lostart = ctr-loctr
    loend = lostart+lodims
    log_rad = log_rad[int(lostart[0]):int(loend[0]),int(lostart[1]):int(loend[1])]
    angle = angle[int(lostart[0]):int(loend[0]),int(lostart[1]):int(loend[1])]
    lodft = lodft[int(lostart[0]):int(loend[0]),int(lostart[1]):int(loend[1])]
    YIrcos = np.abs(np.sqrt(1.0 - Yrcos**2))
    lomask = pointOp(log_rad, YIrcos, Xrcos[0,0], Xrcos[0,1]-Xrcos[0,0])
    lodft = lomask * lodft
    npyr,nind = buildSFpyrLevs(lodft, log_rad, Xrcos, Yrcos, angle, ht-1, nbands)
    pyr = np.concatenate((bands.reshape([-1,1]), npyr.reshape([-1,1])), axis=0)
    pind = np.concatenate((bind, nind), axis=0)
    return pyr, pind


def buildSFpyr(im, ht=1, order=3, twidth=1):
    max_ht = math.floor(np.log2(np.min(im.shape))) - 2
    assert ht <= max_ht, "Cannot build pyramid higher than {} levels.".format(max_ht)
    if order > 15 or order < 0:
        print("Warning: ORDER must be an integer in the range [0,15]. Truncating.")
        order = np.min(np.max(order,0),15)
    else:
        order = round(order)
    nbands = order + 1
    if twidth <= 0:
        print("Warning: TWIDTH must be positive.  Setting to 1.")
        twidth = 1
    if (nbands % 2 == 0):
        harmonics = np.arange(nbands/2).reshape([-1,1])*2 + 1
    else:
        harmonics = np.arange(nbands/2).reshape([-1,1])*2
    steermtx = steer2HarmMtx(harmonics, \
                    np.pi*np.arange(nbands).reshape([1,-1])/nbands, 'even')
    dims = np.array(im.shape)
    ctr = np.ceil((dims+0.5)/2)
    A = (np.arange(1,dims[1]+1)-ctr[1])/(dims[1]/2)
    [xramp,yramp] = np.meshgrid((np.arange(1,dims[1]+1)-ctr[1])/(dims[1]/2), \
                                (np.arange(1,dims[0]+1)-ctr[0])/(dims[0]/2))
    angle = np.arctan2(yramp,xramp)
    log_rad = np.sqrt(xramp**2 + yramp**2)
    log_rad[int(ctr[0])-1,int(ctr[1])-1] =  log_rad[int(ctr[0])-1,int(ctr[1])-2]
    log_rad  = np.log2(log_rad)
    Xrcos,Yrcos = rcosFn(twidth,(-twidth/2),[0,1])
    Yrcos = np.sqrt(Yrcos)
    YIrcos = np.sqrt(1.0 - Yrcos**2)
    #==
    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0,0], Xrcos[0,1]-Xrcos[0,0])
    imdft = fftshift(fft2(im))  # this could be slightly different from matlab code
    lo0dft =  imdft * lo0mask
    pyr,pind = buildSFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands)
    hi0mask = pointOp(log_rad, Yrcos, Xrcos[0,0], Xrcos[0,1]-Xrcos[0,0])
    hi0dft =  imdft * hi0mask
    hi0 = ifft2(ifftshift(hi0dft))
    pyr = np.concatenate((np.real(hi0.reshape([-1,1])), pyr), axis=0)
    pind = np.concatenate((np.array([hi0.shape]), pind), axis=0)
    return pyr,pind,steermtx,harmonics

def pyrBandIndices(pind,band):
    if band > pind.shape[0] or band < 1:
        error(sprintf('BAND_NUM must be between 1 and number of pyramid bands (%d).',\
            size(pind,1)))
    if pind.shape[1] != 2:
        error('INDICES must be an Nx2 matrix indicating the size of the pyramid subbands')
    ind = 1
    for l in range(1,band):
        ind = ind + np.prod(pind[l-1,:])
    indices = np.arange(ind, ind+np.prod(pind[band-1,:]))
    return indices.astype(np.int32)

def pyrBand(pyr, pind, band):
    res = pyr[pyrBandIndices(pind,band)-1, 0].reshape\
                ([int(pind[band-1,0]), int(pind[band-1,1])])
    return res


def norm_sender_normalized(pyro,pind,Nsc,Nor,parent,neighbor,blSzX,blSzY,nbins):
    guardband = 16
    pyro = np.real(pyro)
    Nband = pind.shape[0]-1
    p = 1
    for scale in range(1,Nsc+1):
        for orien in range(1,Nor+1):
            nband = (scale-1)*Nor+orien+1
            aux = pyrBand(pyro, pind, nband)
            (Nsy,Nsx) = aux.shape
            prnt = parent & (nband < Nband-Nor)
            BL = np.zeros([aux.shape[0],aux.shape[1],1 + prnt])
            BL[:,:,0] = aux
            if prnt:
                auxp = pyrBand(pyro, pind, nband+Nor)
                auxp = np.real(imresize(auxp,np.array(auxp.shape)*2))
                BL[:,:,1] = auxp[:Nsy,:Nsx]
            y=BL
            (nv,nh,nb) = y.shape
            block = np.array([[blSzX, blSzY]])
            #==
            nblv = nv-block[0,0]+1   # Discard the outer coefficients
            nblh = nh-block[0,1]+1   # for the reference (centrral) coefficients (to avoid boundary effects)
            nexp = nblv*nblh;      # number of coefficients considered
            N = np.prod(block) + prnt # size of the neighborhood
            #==
            Ly = (block[0,0]-1)/2        # block(1) and block(2) must be odd!
            Lx = (block[0,1]-1)/2
            if (Ly != math.floor(Ly)) or (Lx != math.floor(Lx)):
                print('Spatial dimensions of neighborhood must be odd!')
            Y = np.zeros([nexp,N]);      # It will be the observed signal (rearranged in nexp neighborhoods)
            # Rearrange observed samples in 'nexp' neighborhoods
            Ly = int(Ly)
            Lx = int(Lx)
            n = 0
            for ny in range(-Ly,Ly+1):  # spatial neighbors
                for nx in range(-Lx,Lx+1):
                    n = n + 1
                    #foo = shift(y[:,:,0],[ny nx])
                    foo = np.roll(y[:,:,0], ny, axis=0)
                    foo = np.roll(foo, nx, axis=1)
                    foo = foo[Ly:Ly+nblv,Lx:Lx+nblh]
                    Y[:,n-1:n] = foo.reshape([-1,1])
            if prnt:    #parent
                n = n + 1
                foo = y[:,:,1]
                foo = foo[Ly:Ly+nblv,Lx:Lx+nblh]
                Y[:,n-1:n] = foo.reshape([-1,1])
            if neighbor:
                for neib in range(1,Nor+1):
                    if neib == orien:
                        continue
                    n=n+1
                    nband1 = (scale-1)*Nor+neib+1   # except the ll
                    aux1 = pyrBand(pyro, pind, nband1)
                    aux1 = aux1[Ly:Ly+nblv,Lx:Lx+nblh]
                    Y[:,n-1:n] = aux1.reshape([-1,1])
            C_x = np.dot(Y.T*Y)/nexp

    return subband, size_band

# Input: img already transfered into double, y channel shaved
def global_gsm(im):
    num_or = 6
    num_scales = 2
    pyr,pind, _, _ = buildSFpyr(im, ht=num_scales, order=num_or-1)
    subband, size_band = norm_sender_normalized(pyr,pind,num_scales,num_or,1,1,3,3,50)

