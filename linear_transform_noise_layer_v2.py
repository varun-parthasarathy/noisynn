import torch
import torch.nn.functional as F


def _compute_quality_matrix(k, alpha):
    if alpha is None:
        Q = (torch.diag(torch.ones(k)) * -k/(k+1)) + (torch.ones(k, k) / (k+1))
    else:
        I = torch.diag(torch.ones(k))
        fI = torch.zeros(k, k)
        for i in range(k):
            fI[(i+1)%k, i] = 1 # since f(I) is a cyclic row shift operation
        Q = (-alpha * I) + (alpha * fI)
    return Q

class LinearTransformNoiseLayerV2(torch.nn.Module):
    def __init__(self, batch_size, alpha=None, metric='L2', simple=True, **kwargs):
        '''
        Arguments -
            batch_size - Specify the expected batch size during training. This will be used to generate the quality
                         matrix. If the batch size can change during training, this layer will cause the training
                         loop to fail due to a mismatch in dimensions, in which case you should use V1 instead.
            alpha  -     Sets the linear transform strength. If None, the optimal quality matrix will be computed
            metric -     Sets the distance metric to be used for generating negative pairs for each input using a
                         heuristic approach. Possible values are ['L1', 'L2', 'cosine'] or int. If an integer is
                         passed, the corresponding p-norm will be computed.
            simple -     Use simplified method of getting a negative pair - instead of randomly sampling from the 
                         top n largest distances, take the largest distance only. Can help speed up computations
        
        This implementation of the noise layer assumes a fixed batch size and uses the simplified method of negative
        pair mining by default for speedups during training. If you want to use variable batch sizes, use the V1 of
        this layer instead.
        '''
        super().__init__(**kwargs)
        self.alpha = alpha
        if metric == 'L2':
            self.norm = 2
        elif metric == 'L1':
            self.norm = 1
        else:
            self.norm = metric if isinstance(metric, int) or metric == 'cosine' else 2
        self.simple = simple
        self.batch_size = batch_size
        self.Q = _compute_quality_matrix(self.batch_size, self.alpha)
    
    def generate_random_sample(self, x):
        '''
        Shuffle the batch X to pair each image with one from a different class using the distance between latent space
        representations. The assumption is that for a classification task, the latent space representation (especially
        in deeper layers) for images of the same class would be close to each other compared to representations of images
        from different classes. Note that for this to work effectively, each batch must 
            a. be as large as possible (as more data in the batch reduces task entropy to a greater extent)
            b. have a good mix of images from different classes
        If two images of the same class have a large distance, one of them could be getting mis-classified and thus adding
        features from that image could force the classifier to learn even better features than before, which could still
        improve performance. This is not guaranteed, which is why this approach is a heuristic.

        The benefit of this heuristic is that labels are not required in the forward pass. Thus this layer can be dropped
        into any model with minimal code changes.
        '''
        if len(x.shape) > 2:
            X2 = torch.flatten(x, 1)
        else:
            X2 = x
        if self.norm == 'cosine':
            dists = 1. - (F.normalize(X2) @ F.normalize(X2).t())
        else:
            dists = torch.cdist(F.normalize(X2, p=self.norm), F.normalize(X2, p=self.norm), p=self.norm)

        if self.simple is False:
            k = max(min(x.shape[0], 8), x.shape[0] // 4) # Arbitrary value for top-k
            _, largest_k_indices = torch.topk(dists, k)
            rand_idxs = largest_k_indices[range(x.shape[0]), [torch.randperm(k)[0] for i in range(x.shape[0])]]
            rand_x = x[rand_idxs]
            return rand_x
        else:
            largest_indices = torch.argmax(dists, dim=1)
            rand_x = x[largest_indices]
            return rand_x
        
    def forward(self, x):
        # if self.training:
        #     # Add noise during training only
        #     input_shape = x.shape
        #     # Flatten x for matrix multiplication; axes are just abstracts anyway
        #     x = torch.flatten(x, 1)
        #     Q = self.compute_quality_matrix(input_shape[0])
        #     X2 = self.generate_random_sample(x)
        #     # Add the linear transform noise
        #     x = x + Q@X2
        #     x = x.reshape(input_shape)
        #     return x
        # else:
        #     # During inference, operate as an identity layer
        #     return x

        # During the review process, the authors have mentioned that noise should be added
        # in both training and testing phases, which is a bit weird; however I will be editing
        # this layer to match them for now
        input_shape = x.shape
        x = torch.flatten(x, 1)
        X2 = self.generate_random_sample(x)
        x = x + self.Q@X2
        x = x.reshape(input_shape)
        return x
        

if __name__ == '__main__':
    noise_layer = LinearTransformNoiseLayerV2(16)
    inputs = torch.rand((16, 64, 4, 4))
    outputs = noise_layer(inputs)
    print(outputs.shape, outputs.requires_grad)

    # Grad flow check
    test_model = torch.nn.Sequential(torch.nn.Linear(512, 256), noise_layer, torch.nn.Linear(256, 128), torch.nn.Linear(128, 8))
    inputs = torch.rand((16, 512))
    outputs = test_model(inputs)
    outputs.sum().backward()
    print(test_model[0].weight.grad, test_model[2].weight.grad, test_model[3].weight.grad)

    # Test eval mode
    noise_layer.eval()
    outputs = noise_layer(inputs)
    print(torch.all(outputs == inputs))