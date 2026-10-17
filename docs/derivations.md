# Backpropagation derivation

Backpropagation is the chain rule applied backwards throughout the network.
## 1. Scalar network

Forward pass:

$$
z_1 = w_1 x
$$

$$
a_1 = \operatorname{ReLU}(z_1)
$$

$$
z_2 = w_2 a_1
$$

$$
L = (z_2 - y)^2
$$

ReLU and its derivative:

$$
\operatorname{ReLU}(z) = \max(0, z)
\qquad
\operatorname{ReLU}'(z) =
\begin{cases}
1 & z > 0 \\
0 & z < 0
\end{cases}
$$

np.maximum(0, z) is just what ReLU is, it removes negative values. The derivative is undefined at exactly 0, and the implementation uses 0 there.

### Gradient with respect to $w_2$

$$
\frac{dL}{dw_2} = \frac{dz_2}{dw_2} \times \frac{dL}{dz_2} = a_1 \times 2(z_2 - y)
$$

### Gradient with respect to $w_1$

$$
\frac{dL}{dw_1}
= \frac{dz_1}{dw_1} \times \frac{dL}{dz_2} \times \frac{dz_2}{da_1} \times \frac{da_1}{dz_1}
= x \times 2(z_2 - y) \times w_2 \times \operatorname{ReLU}'(z_1)
$$

If $z_1 < 0$, then $\operatorname{ReLU}'(z_1) = 0$ and the whole gradient is 0. No learning signal reaches there in this case if its zero (dead neuron)