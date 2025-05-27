import torch
import torch.nn as nn
from scipy.linalg import svd
from ops.im2col import Cube2Col,Col2Cube,Im2Col
from collections import namedtuple
from ops.utils import soft_threshold,sparsity,kronecker,Init_DCT
from tqdm import tqdm
from pysptools.material_count import HySime
STARNetParams = namedtuple('STARNetParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings','threshold', 'multi_lmbda','verbose', 'beta', 'mu', 'e', 'gamma_1', 'gamma_2', 'l'])
import  numpy as np
import tensorly as tl
#import t_tools
from skimage.restoration import  denoise_nl_means,estimate_sigma
class STARNet(nn.Module):
    def __init__(self, params: SMDSNetParams):   #初始化
        super(STARNet, self).__init__()
        D=[]
        D.append(Init_DCT(params.kernel_size, params.num_filters[0]))
        D.append(Init_DCT(params.kernel_size, params.num_filters[1]))
        D.append(Init_DCT(params.kernel_size, params.num_filters[2]))
        A=[]
        B=[]
        W=[]
        self.apply_A=nn.ParameterList()
        self.apply_D=nn.ParameterList()
        self.apply_W=nn.ParameterList()
        Dic = kronecker(kronecker(D[0], D[1]), D[2])
        self.dom = torch.pinverse(Dic)
        for i in range(3):
            dtd = D[i].t() @ D[i]
            _, s, _ = dtd.svd()
            l = torch.max(s)
            D[i] /= torch.sqrt(l)
            A.append(D[i].transpose(0, 1))
            B.append(torch.clone(A[i].transpose(0, 1)))
            W.append(torch.clone(A[i].transpose(0, 1)))
            self.apply_A.append(nn.Parameter(A[-1]))
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
            
        self.e = nn.Parameter(torch.zeros(1,1,56))
            

        
        self.mu = nn.Parameter(torch.zeros(1,1, 1,1,total_filters))
        nn.init.constant_(self.mu, params.mu)
        
        self.l = nn.Parameter(torch.zeros(1,1, 1,1,total_filters))
        nn.init.constant_(self.l, params.l)
            
        self.soft_threshold = soft_threshold
        
        
    def forward(self, I):

        R, Ek, I_sub = self.pro_sub(I)

        output = self.denoise_sub(I_sub)

        bs=len(R)
        im=torch.Tensor([]).to(device=I.device)
        for i in range(bs):
            im_sub=output[i].permute([0, 2, 3, 1])
            _im=torch.matmul(im_sub, Ek[i].T)
            im=torch.cat((im,_im),0)
        output=im.permute([0,3,1,2])

        return output

    def pro_sub(self,I):
        hs = HySime()
        bands=I.shape[1]
        R=[]
        Ek=[]
        I_sub=[]
        sigma_est=0
        for _I in I:
            _I=_I.permute([1, 2, 0])
            sigma_est = estimate_sigma(_I.cpu().numpy(), multichannel=True, average_sigmas=True)
            I_nlm=denoise_nl_means(_I.cpu().numpy(), patch_size=7, patch_distance=9, h=0.08, multichannel=True,
                             fast_mode=True,sigma=sigma_est)
            _R, _Ek = hs.count(I_nlm)

            _Ek = torch.FloatTensor(_Ek).to(device=_I.device)
            if _R < self.params.kernel_size:
                _Ek = torch.cat((_Ek, torch.zeros([bands, self.params.kernel_size - _R],dtype=_Ek.dtype).to(device=_I.device)),1)  #确保 _Ek 的宽度与卷积核大小相匹配
            _Ek=_Ek.to(device=_I.device)

            I_sub.append(torch.matmul(_I, _Ek).permute([2,0,1]))
            R.append(_R)
            Ek.append(_Ek)

        return R, Ek, I_sub

    def denoise_sub(self,I):

        params = self.params
        thresh_fn = self.soft_threshold
        bs = len(I)
        I_col=torch.Tensor([])
        batch_ind=[]
        I_size=[]

        L = []
        for _I in I:
            padding = (params.stride - (_I.shape[2] - params.kernel_size) % params.stride) % params.stride

            _I = thresh_fn(_I, self.e)
            im=Cube2Col(_I.unsqueeze(0), kernel_size=params.kernel_size, stride=params.stride, padding=padding, tensorized=True)

            I_size.append([im.shape[2]-1+params.kernel_size,_I.shape[1],_I.shape[2]])
            batch_ind.append(im.shape[2])

            I_col= torch.cat((I_col,im),2)

        I_col=I_col.to(_I.device)
        mean_patch = I_col.mean(dim=1, keepdim=True)
        I_col = I_col - mean_patch

        if I_col.is_cuda:
            self.apply_A = self.apply_A.cuda()
            self.apply_D = self.apply_D.cuda()
            self.apply_W = self.apply_W.cuda()
            self.dom=self.dom.cuda()
        kr_A = kronecker(kronecker(self.apply_A[0], self.apply_A[1]), self.apply_A[2])
        kr_D = kronecker(kronecker(self.apply_D[0], self.apply_D[1]), self.apply_D[2])
        kr_W = kronecker(kronecker(self.apply_W[0], self.apply_W[1]), self.apply_W[2])
        I_col = I_col.permute([0, 2, 3, 4, 1])
        I_col = thresh_fn(I_col, self.mu)

        P = torch.zeros_like(I_col)
        gamma_k=thresh_fn(torch.matmul(I_col,self.dom.t()),self.l)

        num_unfoldings = params.unfoldings
        N = I_col.shape[1] * I_col.shape[2] * I_col.shape[0]
        for k in range(num_unfoldings):

            x_k = torch.matmul(gamma_k, kr_D.t())
            res = x_k - I_col
            r_k = torch.matmul(res, kr_A.t())
            lmbda_ = self.lmbda[k] if params.multi_lmbda else self.lmbda
            beta_ = self.beta[k] if params.multi_lmbda else self.beta
            gamma_1_ = self.gamma_1[k] if params.multi_lmbda else self.gamma_1
            gamma_2_ = self.gamma_2[k] if params.multi_lmbda else self.gamma_2
            gamma_k = thresh_fn(gamma_k - r_k, gamma_1_*lmbda_)

            m = gamma_k.shape[0]
            n = gamma_k.shape[1]

            for i in range (m-1):
                a = gamma_k[i, :, :, :, :] - P[i, :, :, :, :]/beta_
                gamma_k1 = gamma_k
                #print(a.shape)
                for p in range (n-1):
                    gamma_k = t_svd(a[p, :, :, :], gamma_2_*lmbda_/beta_)

                P[i, :, :, :, :] = P[i, :, :, :, :] - beta_*(gamma_k[i, :, :, :, :] - gamma_k1)

            if params.verbose:
                residual = 0.5 * (x_k - I_col).pow(2).sum() / N
                loss = residual + (lmbda_ * gamma_k).abs().sum() / N
                tqdm.write(
                    f'patch res {residual.item():.2e} | sparsity {sparsity(gamma_k):.2f} | loss {loss.item():0.4e} ')

        output_all = torch.matmul(gamma_k, kr_W.t())

        output_all = output_all.permute([0, 4, 1, 2, 3])

        output_all = output_all + mean_patch

        inds=np.cumsum(batch_ind)
        inds=np.hstack(([0],inds))
        output=[]
        for i in range(bs):
            output.append(Col2Cube(output_all[:,:,inds[i]:inds[i+1]], I_size[i], kernel_size=params.kernel_size, stride=params.stride, padding=0,
                              avg=True, input_tensorized=True))
        if params.verbose:
            tqdm.write('')

        return output



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

    return  Ahat




