import torch
import numpy as np
from timm.models.vision_transformer import VisionTransformer

def quality_matrix(k, alpha=None):
    if alpha is None:
        Q = (torch.diag(torch.ones(k)) * -k/(k+1)) + (torch.ones(k, k) / (k+1))
    else:
        I = torch.diag(torch.ones(k))
        fI = torch.zeros(k, k)
        for i in range(k):
            fI[(i+1)%k, i] = 1
        Q = (-alpha * I) + (alpha * fI)
    return Q

class NoisyViT(VisionTransformer):
    def __init__(self, alpha=None, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    @torch.no_grad()
    def generate_linear_transform_noise(self, x, y, B):
        # @TODO: Check if the distance between embeddings can be used to generate the shuffled X data?
        y = torch.squeeze(y) if isinstance(y, torch.Tensor) else torch.squeeze(torch.tensor(y))
        noise_idxs = []
        for c in range(B):
            cur_y = y[c]
            neg_idxs = torch.where(y != cur_y, True, False).nonzero(as_tuple=True)[0]
            neg_id = np.random.choice(neg_idxs)
            noise_idxs.append(neg_id)
        rand_x = x.clone().detach() # This is noise; we do not want to have this in the graph
        return rand_x[noise_idxs]
    
    def forward_features(self, inp):
        if isinstance(inp, tuple):
            x, y = inp
        else:
            x = inp
        if self.grad_checkpointing and not torch.jit.is_scripting():
            return super().forward_features(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # if self.training:
        #     # Add noise only in the training phase
        #     x = self.blocks[:-1](x)
        #     # Assume that the first embedding is the token. We do not want to take the token into account
        #     # when adding noise
        #     token = x[:, 0, :].unsqueeze(1)
        #     x = x[:, 1:, :]
        #     B, L, C = x.shape
        #     # Flatten for matmul purposes, axes are just abstracts anyway
        #     x = torch.flatten(x, 1)

        #     # The dimensions of the quality matrix are batch_size x batch_size as described in the paper. This is
        #     # because the noise itself is the result of linear transformations on the input at the image level
        #     with torch.no_grad(): # For safety
        #         Q = quality_matrix(B, self.alpha)
        #         # Since image X1 needs to receive noise from a different image X2, we sample the negative pairs using labels y
        #         # The benefit is that we do not need to perform any separate sampling strategy from the dataset, which allows
        #         # this to be dropped into any backbone in theory. The issue however is that labels need to be passed in the
        #         # forward pass during training.
        #         rand_x = self.generate_linear_transform_noise(x, y, B)
        #         # Compute X1 + QX2
        #         x = x + Q@rand_x
            
        #     x = x.reshape(B, L, C)
        #     x = torch.cat([token, x], dim=1)
        #     x = self.blocks[-1](x)
        # else:
        #     x = self.blocks(x)

        # During the review process, the authors have mentioned that noise should be added
        # in both training and testing phases, which is a bit weird; however I will be editing
        # this code to match them for now
        x = self.blocks[:-1](x)
        token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]
        B, L, C = x.shape
        x = torch.flatten(x, 1)

        with torch.no_grad(): # For safety
            Q = quality_matrix(B, self.alpha)
            rand_x = self.generate_linear_transform_noise(x, y, B)
            x = x + Q@rand_x
        
        x = x.reshape(B, L, C)
        x = torch.cat([token, x], dim=1)
        x = self.blocks[-1](x)
        x = self.norm(x)
        return x
    

# Sanity check
if __name__ == '__main__':
    model = NoisyViT(alpha=None,
                     patch_size=16,
                     embed_dim=64,
                     depth=12,
                     num_heads=4)
    inputs = torch.rand((16, 3, 224, 224))
    labels = [82, 82, 3, 2, 4, 1, 1, 5, 6, 7, 8, 15, 17, 241, 99, 62]
    output = model((inputs, labels))
    print(output.shape)