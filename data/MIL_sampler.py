from numpy.core.fromnumeric import cumsum
from numpy.random import permutation
import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
from typing import Iterator, Optional, Sequence
from torch import Tensor
import numpy as np

#sampler ensuring all patches in a batch are from the same slide, for 'batch' MIL
class WeightedMILSampler(Sampler[int]):
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, sizes=None, btch_size=None, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.sizes=sizes
        self.nslides=len(self.sizes)
        self.btch_size=btch_size
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_list, csum=[],[0]
        csum.extend(cumsum(self.sizes))  
        batches_per_slide=np.floor(self.num_samples/self.btch_size)
        for k in range(int(batches_per_slide)):
            perm=permutation(range(self.nslides))
            for i in perm:
                rand_tensor = torch.multinomial(self.weights[csum[i]:csum[i+1]], self.btch_size, self.replacement, generator=self.generator)
                rand_list.extend((rand_tensor+csum[i]).tolist())   #do i really want to sample same number per slide?
                #patches from TMA with fewer patches may be over-represented. But otherwise particular slides are under-represented.

        return iter(rand_list)

    def __len__(self) -> int:
        return self.num_samples*self.nslides


class WeightedRandomStratifiedSampler(WeightedRandomSampler):
    #weighted random sampler with additional logic to enforce 
    #sampling of same number from each slide
    def __init__(self, weights: Sequence[float], num_samples: int, replacement: bool = True, sizes=None, generator=None, same_per_slide=True) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.generator = generator
        self.sizes=sizes
        self.nslides=len(self.sizes)
        self.same_per_slide=same_per_slide
        if num_samples<0:
            #max based MIL
            self.num_samples=self.nslides
            self.samples_per_slide=1    
        else:
            if same_per_slide:
                self.samples_per_slide=self.num_samples//self.nslides
                self.num_samples=self.samples_per_slide*self.nslides   #samples per slide
            else:
                self.samples_per_slide=np.floor(sizes*num_samples/sum(sizes)).astype('uint64')
                self.samples_per_slide[self.samples_per_slide==0]=1
                self.num_samples=sum(self.samples_per_slide).astype('uint64')
        

    def __iter__(self) -> Iterator[int]:
        rand_list, csum=[],[0]
        csum.extend(cumsum(self.sizes))
        for i in range(self.nslides):
            if self.same_per_slide:
                n_samp=self.samples_per_slide
            else:
                n_samp=self.samples_per_slide[i]
            if self.num_samples==self.nslides:
                rand_tensor=np.argmax(self.weights[csum[i]:csum[i+1]])
                rand_list.extend([rand_tensor.item()+csum[i]])
            else:
                rand_tensor = torch.multinomial(self.weights[csum[i]:csum[i+1]], n_samp, self.replacement, generator=self.generator)
                rand_list.extend((rand_tensor+csum[i]).tolist())   #do i really want to sample same number per slide?

        rand_list=permutation(rand_list)
        return iter(rand_list)
    