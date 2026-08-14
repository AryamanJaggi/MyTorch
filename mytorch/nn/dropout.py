import numpy as np

class Dropout:
    """
    Inverted dropout: zeroes out each activation independently with
    probability p during training, and scales the survivors by 1/(1-p)
    so that no rescaling is needed at inference time.
    
    """

    def __init__(self, p=0.5):
        self.p = p

    def forward(self, Z, eval=False):
        if eval:
            # Inference mode: dropout is a no-op.
            self.mask = np.ones(Z.shape, dtype="f")
            self.A = Z
        else:
            # Training mode: build a random binary mask, keeping each unit
            # with probability (1-p), then rescale by 1/(1-p) (inverted dropout).
            self.mask = (np.random.rand(*Z.shape) > self.p).astype("f")
            self.A = (Z * self.mask) / (1 - self.p)

        return self.A

    def backward(self, dLdA):
        # Gradient only flows through the units that survived forward,
        # scaled the same way the forward pass was.
        dLdZ = (dLdA * self.mask) / (1 - self.p)
        return dLdZ