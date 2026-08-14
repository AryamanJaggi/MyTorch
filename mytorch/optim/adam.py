import numpy as np


class Adam:
    """
    Adam optimizer

    Maintains per-parameter running estimates of the first moment (mean, m)
    and second moment (uncentered variance, v) of the gradients, with
    bias-correction, and optional L2 weight decay.
    """

    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        # Only layers with weights (e.g. Linear) get updated; skip activation
        # layers, Dropout, etc. that don't have W/b.
        self.l = [layer for layer in model.layers if hasattr(layer, "W")]
        self.L = len(self.l)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # Timestep, used for bias correction.
        self.t = 0

        # First moment (m) and second moment (v) estimates for weights and biases.
        self.m_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.m_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        self.t += 1

        for i in range(self.L):
            dLdW = self.l[i].dLdW
            dLdb = self.l[i].dLdb

            # L2 regularization: add weight_decay * W to the weight gradient.
            # (Biases are conventionally left out of weight decay.)
            if self.weight_decay != 0.0:
                dLdW = dLdW + self.weight_decay * self.l[i].W

            # Update biased first moment estimate.
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dLdW
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * dLdb

            # Update biased second moment estimate.
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dLdW ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (dLdb ** 2)

            # Bias-corrected moment estimates.
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Parameter update.
            self.l[i].W = self.l[i].W - self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            self.l[i].b = self.l[i].b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)