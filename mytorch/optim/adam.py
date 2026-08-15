import numpy as np


class Adam:
    """
    Adam optimizer. 
    Maintains per-parameter running estimates of the first moment (mean, m)
        and second moment (uncentered variance, v) of the gradients, with
        bias-correction, and optional L2 weight decay.
    
    Generalized to work across every parameter-naming
    convention used in this library:
      - Linear-style:   .W / .b       (weight + bias)
      - Norm-style:      .BW / .Bb     (scale + shift, e.g. LayerNorm, BatchNorm1d)
      - Embedding-style: .W only       (weight table, no bias)

    """

    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # Flattened per-parameter bookkeeping: parallel lists, one entry
        # per learnable array (not per layer -- a Linear layer contributes
        # two entries, W and b).
        self._owner = []      # the layer object that owns this parameter
        self._param_attr = [] # attribute name of the parameter itself, e.g. "W"
        self._grad_attr = []  # attribute name of its gradient, e.g. "dLdW"
        self._decay = []      # whether weight decay applies to this parameter

        for layer in model.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                # Linear-style: weight + bias
                self._register(layer, "W", "dLdW", decay=True)
                self._register(layer, "b", "dLdb", decay=False)
            elif hasattr(layer, "BW") and hasattr(layer, "Bb"):
                # Norm-style: scale + shift. Conventionally not weight-decayed.
                self._register(layer, "BW", "dLdBW", decay=False)
                self._register(layer, "Bb", "dLdBb", decay=False)
            elif hasattr(layer, "W"):
                # Embedding-style: weight table only, no bias.
                self._register(layer, "W", "dLdW", decay=True)

        self.L = len(self._owner)
        self.m = [np.zeros_like(getattr(o, a)) for o, a in zip(self._owner, self._param_attr)]
        self.v = [np.zeros_like(getattr(o, a)) for o, a in zip(self._owner, self._param_attr)]

    def _register(self, layer, param_attr, grad_attr, decay):
        self._owner.append(layer)
        self._param_attr.append(param_attr)
        self._grad_attr.append(grad_attr)
        self._decay.append(decay)

    def step(self):
        self.t += 1

        for i in range(self.L):
            owner = self._owner[i]
            p_attr = self._param_attr[i]
            g_attr = self._grad_attr[i]

            P = getattr(owner, p_attr)
            G = getattr(owner, g_attr)

            if self._decay[i] and self.weight_decay != 0.0:
                G = G + self.weight_decay * P

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * G
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (G ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            new_P = P - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(owner, p_attr, new_P)