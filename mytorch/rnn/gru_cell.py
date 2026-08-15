import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Parameters
        ----------
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        # Cache this — it's r_t's "partner" term, needed twice in backward
        # (once for dr, once for dq/dWnh/dbnh/dh_prev).
        self.q = self.Wnh @ self.hidden + self.bnh

        self.r = self.r_act.forward(self.Wrx @ self.x + self.brx + self.Wrh @ self.hidden + self.brh)
        self.z = self.z_act.forward(self.Wzx @ self.x + self.bzx + self.Wzh @ self.hidden + self.bzh)
        self.n = self.h_act.forward(self.Wnx @ self.x + self.bnx + self.r * self.q)
        h_t = (1 - self.z) * self.n + self.z * self.hidden

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Parameters
        ----------
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # --- Step 1: ht = (1 - zt) * nt + zt * h_prev ---
        dz = delta * (self.hidden - self.n)          # dL/dz_t
        dn = delta * (1 - self.z)                     # dL/dn_t
        dh_prev = delta * self.z                      # direct path h_prev -> h_t

        # --- Step 2: nt = tanh(Wnx x + bnx + r * q), q = Wnh h_prev + bnh ---
        dn_pre = self.h_act.backward(dn)               # dL/d(pre-tanh of n)

        self.dWnx = np.outer(dn_pre, self.x)
        self.dbnx = dn_pre

        dr = dn_pre * self.q                            # dL/dr_t (via n)
        dq = dn_pre * self.r                             # dL/dq

        self.dWnh = np.outer(dq, self.hidden)
        self.dbnh = dq
        dh_prev += self.Wnh.T @ dq
        dx = self.Wnx.T @ dn_pre

        # --- Step 3: zt = sigmoid(Wzx x + bzx + Wzh h_prev + bzh) ---
        dz_pre = self.z_act.backward(dz)

        self.dWzx = np.outer(dz_pre, self.x)
        self.dbzx = dz_pre
        self.dWzh = np.outer(dz_pre, self.hidden)
        self.dbzh = dz_pre
        dh_prev += self.Wzh.T @ dz_pre
        dx += self.Wzx.T @ dz_pre

        # --- Step 4: rt = sigmoid(Wrx x + brx + Wrh h_prev + brh) ---
        dr_pre = self.r_act.backward(dr)

        self.dWrx = np.outer(dr_pre, self.x)
        self.dbrx = dr_pre
        self.dWrh = np.outer(dr_pre, self.hidden)
        self.dbrh = dr_pre
        dh_prev += self.Wrh.T @ dr_pre
        dx += self.Wrx.T @ dr_pre

        dh_prev_t = dh_prev

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t