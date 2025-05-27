import torch
import torch.nn as nn
from scipy.linalg import svd
from ops.im2col import Cube2Col,Col2Cube,Im2Col
from collections import namedtuple
from ops.utils import soft_threshold,sparsity,kronecker,Init_DCT
from tqdm import tqdm
from pysptools.material_count import HySime
STARNetParams = namedtuple('STARNetParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings','threshold', 'multi_lmbda','verbose', 'beta', 'gamma_1', 'gamma_2', 'l'])
import  numpy as np
import tensorly as tl
#import t_tools
from skimage.restoration import  denoise_nl_means,estimate_sigma
class STARNet(nn.Module):
    def __init__(self, params: STARNetParams):   #初始化
        super(STARNet, self).__init__()
        D=[]
        D.append(Init_DCT(params.kernel_size, params.num_filters[0]))   #生成字典
        D.append(Init_DCT(params.kernel_size, params.num_filters[1]))
        D.append(Init_DCT(params.kernel_size, params.num_filters[2]))
        A=[]
        B=[]
        W=[]
        self.apply_A=nn.ParameterList()    #参数C
        self.apply_D=nn.ParameterList()
        self.apply_W=nn.ParameterList()
        Dic = kronecker(kronecker(D[0], D[1]), D[2])  #1 D1 ×2 D2 ×3 D3
        self.dom = torch.pinverse(Dic)   #求伪逆
        for i in range(3):
            dtd = D[i].t() @ D[i]   #矩阵 D[i] 的转置乘以自身的结果 1 DT1*D1  2 DT2 D2  3 DT3 D3
            _, s, _ = dtd.svd()   #SVD
            l = torch.max(s)        #最大奇异值
            D[i] /= torch.sqrt(l)  #字典矩阵归一化
            A.append(D[i].transpose(0, 1))   #转置
            B.append(torch.clone(A[i].transpose(0, 1)))  #A的转置
            W.append(torch.clone(A[i].transpose(0, 1)))
            self.apply_A.append(nn.Parameter(A[-1]))   #定义可训练参数
            self.apply_D.append(nn.Parameter(B[-1]))
            self.apply_W.append(nn.Parameter(W[-1]))
        self.params = params
        total_filters=params.num_filters[0]*params.num_filters[1]*params.num_filters[2]
        if params.multi_lmbda:
            self.lmbda = nn.ParameterList(
            [nn.Parameter(torch.zeros(1,1, 1,1,total_filters)) for _ in range(params.unfoldings)])
            [nn.init.constant_(x, params.threshold) for x in self.lmbda]
        else:
            self.lmbda = nn.Parameter(torch.zeros(1,1, 1,1,total_filters))
            nn.init.constant_(self.lmbda, params.threshold)
            
        if params.multi_lmbda:
            self.gamma_1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(1,1, 1,1,total_filters)) for _ in range(params.unfoldings)])
            [nn.init.constant_(x, params.threshold) for x in self.gamma_1]
        else:
            self.gamma_1 = nn.Parameter(torch.zeros(1,1, 1,1,total_filters))
            nn.init.constant_(self.gamma_1, params.threshold)            
            
        if params.multi_lmbda:
            self.gamma_2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(1,1, 1,1,total_filters)) for _ in range(params.unfoldings)])
            [nn.init.constant_(x, params.threshold) for x in self.gamma_2]
        else:
            self.gamma_2 = nn.Parameter(torch.zeros(1,1, 1,1,total_filters))
            nn.init.constant_(self.gamma_2, params.threshold)                

        if params.multi_lmbda:
            self.beta = nn.ParameterList(
            [nn.Parameter(torch.zeros(1,1, 1,1,total_filters)) for _ in range(params.unfoldings)])
            [nn.init.constant_(x, params.beta) for x in self.beta]
        else:
            self.beta = nn.Parameter(torch.zeros(1,1, 1,1,total_filters))
            nn.init.constant_(self.beta, params.beta)
            
        self.l = nn.Parameter(torch.zeros(1,1, 1,1,total_filters))
        nn.init.constant_(self.l, params.l)
        #self.beta = params.beta

        self.soft_threshold = soft_threshold
    def forward(self, I):
        #print(I.shape)   # 2 31 9 9
        R, Ek, I_sub = self.pro_sub(I) #_R---signal subspace dimension,子空间维度   _Ek---子空间下的矩阵--A   I_sub--G
        #print(I_sub.shape)
        #list_shape = np.array(I_sub).shape
        #print(list_shape)
        output = self.denoise_sub(I_sub)    #子空间下的图像进行去噪---output--G
        #list_shape = np.array(output).shape
        #print(list_shape)
        bs=len(R)   #子空间维度
        im=torch.Tensor([]).to(device=I.device)
        for i in range(bs):
            im_sub=output[i].permute([0, 2, 3, 1])
            _im=torch.matmul(im_sub, Ek[i].T)  #公式（2）
            im=torch.cat((im,_im),0)
        output=im.permute([0,3,1,2])  #----X
       #print(output.shape)
        return output   # 2 31 9 9

    def pro_sub(self,I):
        hs = HySime()   #高光谱子空间识别算法。子空间识别步骤可以得到高光谱降维后的有效波段
        bands=I.shape[1] #计算了输入张量 I 的第二个维度的大小，并将结果保存在变量 bands 中
        R=[]
        Ek=[]
        I_sub=[]            #子空间下的输入图像
        sigma_est=0
        for _I in I:
            _I=_I.permute([1, 2, 0]) #原先的第二个维度变为第一个维度，第三个维度变为第二个维度，第一个维度变为第三个维度。
            sigma_est = estimate_sigma(_I.cpu().numpy(), multichannel=True, average_sigmas=True) #进行噪声标准差的估计   channel_axis=-1
            I_nlm=denoise_nl_means(_I.cpu().numpy(), patch_size=7, patch_distance=9, h=0.08, multichannel=True,
                             fast_mode=True,sigma=sigma_est) #对输入数据进行非局部均值去噪处理
            _R, _Ek = hs.count(I_nlm)   #Hyperspectral signal subspace estimation  子空间估计
            #_R---signal subspace dimension,子空间维度   _Ek---子空间下的矩阵---A
            _Ek = torch.FloatTensor(_Ek).to(device=_I.device)  #转化为float型
            if _R < self.params.kernel_size:
                _Ek = torch.cat((_Ek, torch.zeros([bands, self.params.kernel_size - _R],dtype=_Ek.dtype).to(device=_I.device)),1)  #确保 _Ek 的宽度与卷积核大小相匹配
            _Ek=_Ek.to(device=_I.device) #确保 _Ek 和 _I 张量在同一设备上进行计算
            #print(torch.matmul(_I, _Ek).permute([2,0,1]).shape)
            I_sub.append(torch.matmul(_I, _Ek).permute([2,0,1]))  #张量乘法，得到子空间下的I  公式(5)--G
            R.append(_R)
            Ek.append(_Ek)
        #arr = np.array(Ek)
        #print(arr.shape)
        #arr = np.array(I_sub)
        #print(arr.shape)
        return R, Ek, I_sub

    def denoise_sub(self,I):    #此处I为子空间下图像

        params = self.params
        thresh_fn = self.soft_threshold
        bs = len(I)
        I_col=torch.Tensor([]) #创建一个空张量
        batch_ind=[]
        I_size=[]
       # P = []
        L = []
        for _I in I:
            padding = (params.stride - (_I.shape[2] - params.kernel_size) % params.stride) % params.stride
            #unsqueeze(0) 在最前面增加一维
            #print(_I.shape)  #9 9 9    ; 10 9 9
            im=Cube2Col(_I.unsqueeze(0), kernel_size=params.kernel_size, stride=params.stride, padding=padding, tensorized=True)
            #print(im.shape) # 1 729 1 1 1  ;  1 729 2 1 1
            I_size.append([im.shape[2]-1+params.kernel_size,_I.shape[1],_I.shape[2]])  #shape[0]就是读取第一维度的长度
            batch_ind.append(im.shape[2])
            #print(I_col.shape) # 0  ;  1 729 1 1 1
            I_col= torch.cat((I_col,im),2)  #将第三维拼接到一起
            #print(I_col.shape) # 1 729 1 1 1  ;  1 729 3 1 1
        #  Extract overlapping cubes from G
        I_col=I_col.to(_I.device)
        mean_patch = I_col.mean(dim=1, keepdim=True)
        I_col = I_col - mean_patch
        #print(I_col.shape)   # 1 3 1 1 729
        if I_col.is_cuda:
            self.apply_A = self.apply_A.cuda()  #将self.apply_A转移到GPU上进行计算
            self.apply_D = self.apply_D.cuda()
            self.apply_W = self.apply_W.cuda()
            self.dom=self.dom.cuda()
        kr_A = kronecker(kronecker(self.apply_A[0], self.apply_A[1]), self.apply_A[2])
        kr_D = kronecker(kronecker(self.apply_D[0], self.apply_D[1]), self.apply_D[2])
        kr_W = kronecker(kronecker(self.apply_W[0], self.apply_W[1]), self.apply_W[2])
        I_col = I_col.permute([0, 2, 3, 4, 1])     #dom --- 1 D1 ×2 D2 ×3 D3的伪逆
        P = torch.zeros_like(I_col)
        #print(I_col.shape) # 1 3 1 1 729
        gamma_k=thresh_fn(torch.matmul(I_col,self.dom.t()),self.l) #soft_threshold 相当于RELU  gamma_k--Bi
        #print(gamma_k.shape)   # 1 3 1 1 729
        num_unfoldings = params.unfoldings
        N = I_col.shape[1] * I_col.shape[2] * I_col.shape[0]
        for k in range(num_unfoldings):    #递归神经网络部分  循环
            #gamma_k = torch.tensor(gamma_k)
            x_k = torch.matmul(gamma_k, kr_D.t())
            res = x_k - I_col   #公式(10)
            r_k = torch.matmul(res, kr_A.t())  #公式(11)第2项
            lmbda_ = self.lmbda[k] if params.multi_lmbda else self.lmbda
            beta_ = self.beta[k] if params.multi_lmbda else self.beta
            gamma_1_ = self.gamma_1[k] if params.multi_lmbda else self.gamma_1
            gamma_2_ = self.gamma_2[k] if params.multi_lmbda else self.gamma_2
            gamma_k = thresh_fn(gamma_k - r_k,  gamma_1_*lmbda_)  #公式(12)  1 3 1 1 729
            #print(gamma_k.shape)
            #print(gamma_k)
            m = gamma_k.shape[0]
            n = gamma_k.shape[1]

            for i in range (m-1):
                a = gamma_k[i, :, :, :, :] - P[i, :, :, :, :]/beta_
                gamma_k1 = gamma_k
                #print(a.shape)
                for p in range (n-1):
                    gamma_k = t_svd(a[p, :, :, :], gamma_2_*lmbda_/beta_)
                #P[i, :, :, :, :] = P[i, :, :, :, :] + self.beta*(gamma_k[i, :, :, :, :] -gamma_k1)
                P[i, :, :, :, :] = P[i, :, :, :, :] - beta_*(gamma_k[i, :, :, :, :] - gamma_k1)
            #gamma_k  = gamma_k .to(device=_I.device)
            #print('111')
            #print(gamma_k.shape)
           # print(gamma_k)
            #gamma_k = torch.tensor(gamma_k)
            #gamma_k = gamma_k.cuda
            #print('222')
            #print(gamma_k.shape)
            if params.verbose:
                residual = 0.5 * (x_k - I_col).pow(2).sum() / N  #平方后求和
                loss = residual + (lmbda_ * gamma_k).abs().sum() / N   #损失函数
                tqdm.write(
                    f'patch res {residual.item():.2e} | sparsity {sparsity(gamma_k):.2f} | loss {loss.item():0.4e} ')

        output_all = torch.matmul(gamma_k, kr_W.t())   # B(K ) i ×1 W1 ×2 W2 ×3 W3
        #print(output_all.shape)  # 1 3 1 1 729
        output_all = output_all.permute([0, 4, 1, 2, 3])
        #print(output_all.shape)  # 1 3 1 1 729
        output_all = output_all + mean_patch
        #print(output_all.shape)  # 1 3 1 1 729
        inds=np.cumsum(batch_ind)  # 进行累积求和
        inds=np.hstack(([0],inds)) #将数组 inds 前面插入一个零
        output=[]
        for i in range(bs):
            output.append(Col2Cube(output_all[:,:,inds[i]:inds[i+1]], I_size[i], kernel_size=params.kernel_size, stride=params.stride, padding=0,
                              avg=True, input_tensorized=True))   #式(8)
        if params.verbose:
            tqdm.write('')

        return output


#tl.set_backend('numpy')


def t_svd(A,lamb):
    Af = np.fft.fft(A, axis=2)
    U = np.zeros(tuple([A.shape[0], A.shape[0], A.shape[2]]), dtype=complex)
    S = np.zeros(tuple(A.shape), dtype=complex)
    V = np.zeros(tuple([A.shape[1], A.shape[1], A.shape[2]]), dtype=complex)

    for i3 in range(A.shape[2]):
        uf, sf, vf = np.linalg.svd(Af[:, :, i3])
        sf = np.diag(sf)

        U[:, :, i3] = uf
        S[:, :, i3] = sf
        V[:, :, i3] = vf
    U = np.fft.ifft(U, axis=2)
    S = np.fft.ifft(S, axis=2)
    V = np.fft.ifft(V, axis=2)
    S = soft_threshold(S, lamb)
    Ahat = torch.matmul(torch.matmul(U, S), V)
    #Ahat = t_tools.t_prod(t_tools.t_prod(U, S), V)

   # return U, S, V, Ahat
    return  Ahat



def tr_svd(tensor, rank):
    # Reshape tensor into a matrix
    tensor = tensor.detach().cpu().numpy()
    shape = tensor.shape
    matrix = np.reshape(tensor, (shape[0], -1))

    # Perform SVD on the matrix
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Truncate to the desired rank
    U_trunc = U[:, :rank]
    s_trunc = np.diag(s[:rank])
    #s_trunc = soft_threshold(s_trunc , rank)
    Vt_trunc = Vt[:rank, :]

    # Compute the approximation
    tensor_approx = np.dot(U_trunc, np.dot(s_trunc, Vt_trunc))

    # Reshape back to original tensor shape
    tensor_approx = np.reshape(tensor_approx, shape)

   # return tensor_approx, U_trunc, s_trunc, Vt_trunc
    return tensor_approx


def prox_tnn_w(Y, rho):
    para = 4
    dim = Y.ndim
    print(Y.shape)
    #print(Y)
    n0, n4, n1, n2, n3 = Y.shape  #1 2 1 1 729
    n12 = min(n2, n3)

    Y_cpu = Y.detach().cpu().numpy()

    Yf = np.fft.fftn(Y_cpu, axes=(dim - 1,))
    Uf = np.zeros((n0, n4, n1, n12, n3), dtype=np.float32)
    Vf = np.zeros((n0, n4, n2, n12, n3), dtype=np.float32)
    Sf = np.zeros((n0, n4, n12, n12, n3), dtype=np.float32)

    Yf[np.isnan(Yf)] = 0
    Yf[np.isinf(Yf)] = 0

    trank = 0
    endValue = int(n3 / 2 + 1)
    for i in range(1, endValue):
        U, S, Vt = svd(Yf[0,0,i-1, :, :].reshape(n2, n3), full_matrices=False)  # Reshape to 2D matrix
        Uf[0,0,i-1, :, :] = U.reshape(n12, n3)  # Reshape U to (n1, n12)
        Vf[0,0,i-1, :, :] = Vt.T.conj()  # Transpose and conjugate to get Vf
        s1 = np.diag(S)

        sikema = 10 ** -6
        diagsj_w = 1. / (s1 + sikema)
        diagsj_w = diagsj_w / diagsj_w[para - 1]

        s = np.maximum(s1 - rho * diagsj_w, 0)

        Sf[:, :, i - 1] = np.diag(s)
        temp = np.sum(s > 0)
        trank = max(temp, trank)

    for j in range(n3, endValue, -1):
        Uf[:, :, j - 1] = np.conj(Uf[:, :, n3 - j + 1])
        Vf[:, :, j - 1] = np.conj(Vf[:, :, n3 - j + 1])
        Sf[:, :, j - 1] = Sf[:, :, n3 - j + 1]

    Uf = Uf[:, :trank, :]
    Vf = Vf[:, :trank, :]
    Sf = Sf[:trank, :trank, :]

    U = np.fft.ifftn(Uf, axes=dim - 1)
    S = np.fft.ifftn(Sf, axes=dim - 1)
    V = np.fft.ifftn(Vf, axes=dim - 1)

    X = np.matmul(np.matmul(U, S), np.transpose(V, axes=(0, 2, 1)))
    tnn = np.sum(np.diag(Sf[:, :, 0]))
    #return X, tnn, trank
    return X

