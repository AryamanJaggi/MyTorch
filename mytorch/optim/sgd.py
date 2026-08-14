import numpy as np


class SGD:

    def __init__(self, model, lr=0.1, momentum=0, weight_decay=0.0):
        # Initialize the model layers.
        # NOTE: model.layers mixes Linear layers with activation layers (e.g. ReLU),
        # but only Linear layers have W/b attributes. Filter to just those layers
        # so we don't crash trying to read .W/.b off an activation layer.
        self.l = [layer for layer in model.layers if hasattr(layer, "W")]
        self.L = len(self.l)

        # Assign learning rate, momentum, and L2 weight decay coefficient.
        self.lr = lr
        self.mu = momentum
        self.weight_decay = weight_decay

        # Initialize momentum terms (velocity) for weights and biases to zeros.
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        # Update weights and biases for each layer.
        for i in range(self.L):
            dLdW = self.l[i].dLdW

            # L2 regularization: add weight_decay * W to the weight gradient.
            # (Biases are conventionally excluded from weight decay.)
            if self.weight_decay != 0.0:
                dLdW = dLdW + self.weight_decay * self.l[i].W

            if self.mu == 0:
                # If no momentum, use standard SGD update.
                self.l[i].W = self.l[i].W - self.lr*dLdW  # Update weights using gradients
                self.l[i].b = self.l[i].b - self.lr*self.l[i].dLdb  # Update biases using gradients
            else:
                # If momentum is used, update velocity terms.
                self.v_W[i] = self.mu*self.v_W[i] + dLdW  # Update weight velocity
                self.v_b[i] = self.mu*self.v_b[i] + self.l[i].dLdb  # Update bias velocity

                # Update weights and biases using momentum and learning rate.
                self.l[i].W = self.l[i].W - self.lr*self.v_W[i]  # Update weights using weight velocity
                self.l[i].b = self.l[i].b - self.lr*self.v_b[i]  # Update biases using bias velocity