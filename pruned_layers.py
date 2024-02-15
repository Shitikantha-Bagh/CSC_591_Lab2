import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.mask = np.ones([self.out_features, self.in_features])
        m = self.in_features
        n = self.out_features
        self.sparsity = 1.0
        # Initailization 
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weights = self.linear.weight.data.view(-1).cpu().numpy()
        threshold = np.percentile(np.abs(weights),q)
        self.mask = (np.abs(weights) >= threshold)
        mask_tensor = torch.from_numpy(self.mask).float().to(self.linear.weight.device)
        mask_tensor = mask_tensor.view_as(self.linear.weight.data)
        self.linear.weight.data*= mask_tensor
        self.sparsity = 1.0 - np.sum (self.mask) / len (weights)
        


    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """

        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weights = self.linear.weight.data.view(-1).cpu().numpy()
        threshold = np.std(weights)*s
        self.mask = np.abs(weights) >= threshold
        mask_tensor = torch.from_numpy(self.mask).float().to(self.linear.weight.device)
        mask_tensor = mask_tensor.view_as(self.linear.weight.data)
        self.linear.weight.data *= mask_tensor
        self.sparsity = 1.0 - np.sum(self.mask) / len (weights)
        

class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Expand and Transpose to match the dimension
        self.mask = np.ones_like([out_channels, in_channels, kernel_size, kernel_size])

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weights = self.conv.weight.data.view(-1).cpu().numpy()
        threshold = np.percentile(np.abs(weights),q)
        self.mask = np.abs(weights) >= threshold
        mask_tensor = torch.from_numpy(self.mask).float().to(self.conv.weight.device)
        mask_tensor = mask_tensor.view_as(self.conv.weight.data)
        self.conv.weight.data*= mask_tensor
        self.sparsity = 1.0 - np.sum (self.mask) / len (weights)

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        weights = self.conv.weight.data.view(-1).cpu().numpy()
        threshold = np.std(weights)*s
        self.mask = np.abs(weights) >= threshold
        mask_tensor = torch.from_numpy(self.mask).float().to(self.conv.weight.device)
        mask_tensor = mask_tensor.view_as(self.conv.weight.data)
        self.conv.weight.data *=mask_tensor
        self.sparsity = 1.0 - np.sum(self.mask) / len (weights)
        

