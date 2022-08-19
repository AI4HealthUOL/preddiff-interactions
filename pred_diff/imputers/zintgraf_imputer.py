import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.segmentation import expand_labels
from tqdm.auto import tqdm
import torchvision.transforms

from .imputer_base import ImputerBase

class TensorTransformDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    
    def __init__(self, tensors, tfms=None) -> None:
        self.tensors = tensors
        self.tfms = tfms

    def __getitem__(self, index):
        return self.tfms(self.tensors[index]) if self.tfms else self.tensors[index]

    def __len__(self):
        return self.tensors.size(0)
        
class ZintgrafImputer(ImputerBase):
    """
    Gaussian Imputer from Zintgraf et al
    """
    def __init__(self, train_data: np.ndarray, border=2, max_samples=0, channel_wise=False,bs=128, patch_size=None, epochs=1, gpu=False, transforms=None, **kwargs):
        #optionally passing mean and cov if precomputed
        super().__init__()
        self.train_data = train_data
        self.imputer_name = 'ZintgrafImputer'
        self.border = border
        self.channel_wise = channel_wise
        self.inpatch_idx, self.outpatch_idx = None, None

        self.minmax = np.min(train_data), np.max(train_data)
        if(transforms is None):
            tfms = torchvision.transforms.RandomCrop(patch_size) if patch_size is not None else None
            #tfms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.RandomCrop(patch_size),torchvision.transforms.ToTensor()]) if patch_size is not None else None
        else:
            tfms = transforms
        tensor_train = TensorTransformDataset(torch.Tensor(train_data),tfms)
        tensor_dl = DataLoader(tensor_train,batch_size=bs) 
        self.patch_size = patch_size

        self.mean, self.cov = self.get_online_mean_cov(tensor_dl,max_samples=max_samples,channel_wise=channel_wise,epochs=epochs,gpu=gpu,patch_size=patch_size)

    @staticmethod
    def get_online_mean_cov(dl_cropped,max_samples=20000,channel_wise=True,epochs=1,gpu=False,patch_size=None,scaling_factor=1000.):
        #dl should provide already appropriately cropped inputs
        for x in dl_cropped:
            assert(patch_size is None or (x.size(2)==patch_size and x.size(3)==patch_size))
            if(gpu):
                x=x.cuda()
            if(channel_wise):
                channels = x.size(1)
                features = np.prod(x.size()[2:])
            else:
                features = np.prod(x.size()[1:])
            break

        if(channel_wise):
            mean_prev = torch.zeros(channels,features).to(x.device)
            mean_cur = torch.zeros(channels,features).to(x.device)
            c = torch.zeros(channels,features,features).to(x.device)#store redundant ignoring symmetry and use broadcasting
        else:
            mean_prev = torch.zeros(features).to(x.device)
            mean_cur = torch.zeros(features).to(x.device)
            c = torch.zeros(features,features).to(x.device)#store redundant ignoring symmetry and use broadcasting

        samples = 0
        with torch.no_grad():
            for i in tqdm(list(range(epochs)),leave=False):
                for x in tqdm(dl_cropped,leave=False):
                    if(gpu):
                        x=x.cuda()
                    x = x/scaling_factor
                    if(channel_wise):
                        x = x.view(x.size(0),channels,-1)
                    else:
                        x = x.view(x.size(0),-1)
                    mean_prev =mean_cur.clone()
                    mean_cur = mean_prev + torch.sum(x-mean_prev.unsqueeze(0),dim=0)/(samples+x.size(0))

                    if(channel_wise):
                        c += torch.sum((x.unsqueeze(3)-mean_prev.unsqueeze(2))*(x.unsqueeze(2)-mean_cur.unsqueeze(1)),dim=0)
                    else:
                        c += torch.sum((x.unsqueeze(2)-mean_prev.unsqueeze(1))*(x.unsqueeze(1)-mean_cur),dim=0)
                    samples += x.size(0)
                    if(max_samples>0 and samples>=max_samples):
                        break
        return mean_cur*scaling_factor, c/samples*scaling_factor*scaling_factor

    @staticmethod
    def get_conditional_aux_vars(mean,cov,inpatch_idxs,outpatch_idxs,channel_wise=True):
        '''for given inpatch_idxs (and mean cov for a specific crop size)'''

        with torch.no_grad():
            if(channel_wise):
                mu1 = mean[:,inpatch_idxs]
                mu2 = mean[:,outpatch_idxs]

                cov1x = cov[:,inpatch_idxs]
                cov2x = cov[:,outpatch_idxs]

                cov11 = cov1x[:,:,inpatch_idxs]
                cov12 = cov1x[:,:,outpatch_idxs]
                cov21 = cov2x[:,:,inpatch_idxs]
                cov22 = cov2x[:,:,outpatch_idxs]
                try:
                    cov22inv = cov22.inverse()
                except:#use pseudoinverse if singular
                    print("Info: singular matrix, using pseudo-inverse")
                    cov22inv = torch.pinverse(cov22)

                dotProdForMean = torch.matmul(cov12,cov22inv)
                cond_cov = cov11 - torch.matmul(dotProdForMean,cov21)
            else:
                mu1 = mean[inpatch_idxs]
                mu2 = mean[outpatch_idxs]

                cov1x = cov[inpatch_idxs]
                cov2x = cov[outpatch_idxs]

                cov11 = cov1x[:,inpatch_idxs]
                cov12 = cov1x[:,outpatch_idxs]
                cov21 = cov2x[:,inpatch_idxs]
                cov22 = cov2x[:,outpatch_idxs]
                try:
                    cov22inv = cov22.inverse()
                except:#use pseudoinverse if singular
                    print("Info: singular matrix, using pseudo-inverse")           
                    cov22inv = torch.pinverse(cov22)

                dotProdForMean = torch.matmul(cov12,cov22inv)
                cond_cov = cov11 - torch.matmul(dotProdForMean,cov21)
        return dotProdForMean, cond_cov

    @staticmethod
    def sample_multivariate_gaussian(mean,cov,num_samples):
        try: 
            A = torch.cholesky(cov)#could be precomputed in set_inpatch_idxs
            z = torch.randn(mean.size(-1),num_samples,device=mean.device)
            print("Info: Cholesky decomposition successful")
            return mean+torch.mm(A,z).transpose(0,1)
            #m = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
            #return torch.cat([m.sample().unsqueeze(0).to(mean.device) for _ in range(num_samples)],dim=0)
        except:#SVD https://stats.stackexchange.com/questions/238655/sampling-from-matrix-variate-normal-distribution-with-singular-covariances
            print("Info: Cholesky decomposition failed- falling back to SVD")
            U, S, _ = torch.svd(cov)
            return mean + torch.randn(num_samples,len(mean)).to(mean.device) @ torch.diag(torch.sqrt(S)) @ U.transpose(0,1)

    def _set_inpatch_idxs(self,  inpatch_idxs, outpatch_idxs):
        self.inpatch_idxs, self.outpatch_idxs = inpatch_idxs, outpatch_idxs
        if(len(self.outpatch_idxs)>0):
            self.dotProdForMean, self.cond_cov = self.get_conditional_aux_vars(self.mean,self.cov,inpatch_idxs,outpatch_idxs,channel_wise=self.channel_wise)

    def sample(self,featvec,num_samples=10):
        assert(self.inpatch_idxs is not None and self.outpatch_idxs is not None)

        if(len(self.outpatch_idxs)==0):#no conditioning
            if(len(self.inpatch_idxs)==len(self.mean)):
                cond_mean = self.mean
                cond_cov = self.cov
            else:
                if(self.channel_wise):
                    cond_mean = self.mean[:,self.inpatch_idxs]
                    cond_cov = (self.cov[:,self.inpatch_idxs])[:,:,self.inpatch_idxs]
                else:
                    cond_mean = self.mean[self.inpatch_idxs]
                    cond_cov = (self.cov[self.inpatch_idxs])[:,self.inpatch_idxs]
        else:#conditional
            if(self.channel_wise):
                cond_mean = self.mean[:,self.inpatch_idxs] + torch.matmul(self.dotProdForMean, (featvec[:,self.outpatch_idxs]-self.mean[:,self.outpatch_idxs]).unsqueeze(2)).squeeze(2)
            else:
                cond_mean = self.mean[self.inpatch_idxs] + torch.matmul(self.dotProdForMean, (featvec[self.outpatch_idxs]-self.mean[self.outpatch_idxs]))
            cond_cov = self.cond_cov

        if(self.channel_wise):
            res = []
            for c in range(cond_mean.size(0)):
                res.append(self.sample_multivariate_gaussian(cond_mean[c],cond_cov[c],num_samples))
            return torch.stack(tuple(res),dim=1)
        else:
            return self.sample_multivariate_gaussian(cond_mean,cond_cov,num_samples)
    
    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        """
        test_data: (1, 3, n_pixel, n_pixel), with 3 rgb channels
        """
        # returns only imputation for a single mask
        assert (1, 3) == test_data.shape[:2] or (1, 1) == test_data.shape[:2], f'incorrect shape, test_data.shape =  {test_data.shape}'
        #remove imputed region just to be sure
        test_data = test_data*(1-mask_impute)
        #prepare return value
        test_data_imputed = np.repeat(test_data,n_imputations,axis=0)
        if(self.channel_wise):
            test_data_imputed = test_data_imputed.reshape(test_data_imputed.shape[0],test_data_imputed.shape[1],-1)
        else:
            test_data_imputed = test_data_imputed.reshape(test_data_imputed.shape[0],-1)
            
        image_shape = test_data.shape[1:]
        #prepare masks
        if(len(mask_impute.shape)>2):
            mask_impute2 = mask_impute.copy().mean(axis=0)
        else:
            mask_impute2 = mask_impute
        #determine inpatch and outpatch ids
        if(self.patch_size is not None):
            #mask_expanded = expand_labels(mask_impute2, distance=self.border)
            c1,c2 = np.where(mask_impute2)
            c1min, c2min, c1max, c2max = np.min(c1), np.min(c2), np.max(c1), np.max(c2)
            c1diff = c1max - c1min
            c2diff = c2max - c2min
            if(c1diff>self.patch_size or c2diff>self.patch_size):
                print("Warning: patch size too small. Imputation will be incomplete")
            startidx1=max(0,c1min-(self.patch_size-c1diff)//2)
            startidx2=max(0,c2min-(self.patch_size-c2diff)//2)
            #patch mask in case the patch size is too small
            mask_patch = np.zeros_like(mask_impute2)
            mask_patch[startidx1:startidx1+self.patch_size,startidx2:startidx2+self.patch_size] = 1
            mask_patch = mask_patch*mask_impute2
            #store inpatch idxs for later
            if(self.channel_wise):
                inpatch_idxs_original = np.where(mask_patch.reshape(-1))[0]
            else:
                inpatch_idxs_original = np.where(np.repeat(np.expand_dims(mask_patch,0),test_data.shape[1]).reshape(-1).reshape(-1))[0]
            #restrict to patchwise data
            test_data = test_data[:,:,startidx1:startidx1+self.patch_size,startidx2:startidx2+self.patch_size]
            mask_impute2 = mask_impute2[startidx1:startidx1+self.patch_size,startidx2:startidx2+self.patch_size]
            
        if(self.channel_wise):
            outpatch_idxs = np.where((expand_labels(mask_impute2, distance=self.border)-mask_impute2).reshape(-1))[0]
            inpatch_idxs = np.where(mask_impute2.reshape(-1))[0]
            test_data2 = test_data.copy().reshape(test_data.shape[1],-1)
        else:
            outpatch_tmp = (expand_labels(mask_impute2, distance=self.border)-mask_impute2).reshape(-1)
            outpatch_idxs = np.where(np.repeat(np.expand_dims(outpatch_tmp,0),test_data.shape[1]).reshape(-1).reshape(-1))[0]
            inpatch_idxs = np.where(np.repeat(np.expand_dims(mask_impute2,0),test_data.shape[1]).reshape(-1).reshape(-1))[0]
            test_data2 = test_data.copy().reshape(-1)
        self._set_inpatch_idxs(inpatch_idxs, outpatch_idxs)
    
        #clear imputation region just to be sure
        if(self.channel_wise):
            test_data2[:,self.inpatch_idxs]=0
        else:
            test_data2[self.inpatch_idxs]=0
        
        imputations = self.sample(torch.tensor(test_data2),n_imputations).cpu().numpy()
        imputations = np.clip(imputations.round(), self.minmax[0], self.minmax[1])

        
        if(self.channel_wise):
            test_data_imputed[:,:,inpatch_idxs if self.patch_size is None else inpatch_idxs_original] = imputations.reshape(test_data_imputed.shape[0],test_data_imputed.shape[1],-1)
        else:
            test_data_imputed[:,inpatch_idxs if self.patch_size is None else inpatch_idxs_original] = imputations
        test_data_imputed = test_data_imputed.reshape(n_imputations,1,image_shape[0],image_shape[1],image_shape[2])
        
        return test_data_imputed, None

