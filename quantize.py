import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weights = m.conv.weight.data.cpu().numpy().flatten()
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            kmeans = KMeans(n_clusters=2**bits,init='random')
            kmeans.fit(weights.reshape(-1,1))
            quantized_weights = kmeans.cluster_centers_[kmeans.predict(weights.reshape(-1,1))].reshape(weights.shape)
            m.conv.weight.data = torch.from_numpy(quantized_weights.reshape(m.conv.weight.data.shape)).to(device)
            cluster_centers.append(kmeans.cluster_centers_)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weights = m.linear.weight.data.cpu().numpy().flatten()
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            kmeans = KMeans(n_clusters=2**bits,init='random')
            kmeans.fit(weights.reshape(-1,1))
            quantized_weights = kmeans.cluster_centers_[kmeans.predict(weights.reshape(-1,1))].reshape(weights.shape)
            m.linear.weight.data = torch.from_numpy(quantized_weights.reshape(m.linear.weight.data.shape)).to(device)
            cluster_centers.append(kmeans.cluster_centers_)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

