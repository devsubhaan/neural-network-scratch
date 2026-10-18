# Backpropagation derivation
 
Backpropagation is the chain rule applied backwards through the network.
 
The scalar and vector sections use squared error because the algebra is simple. The softmax section swaps in cross-entropy, which only changes the first backward step. Every step after it is identical.
 
## Scalar network
 
Forward pass:
 
$$
z_1 = w_1 x
$$
 
$$
a_1 = \mathrm{ReLU}(z_1)
$$
 
$$
z_2 = w_2 a_1
$$
 
$$
L = (z_2 - y)^2
$$
 
ReLU and its derivative:
 
$$
\mathrm{ReLU}(z) = \max(0, z)
$$
 
$$
\mathrm{ReLU}'(z) =
\begin{cases}
1 & z > 0 \\
0 & z < 0
\end{cases}
$$
 
`np.maximum(0, z)` implements ReLU: it removes negative values. The derivative is undefined at exactly 0, and the code uses 0 there.
 
### Gradient with respect to $w_2$
 
$$
\frac{dL}{dw_2}
=
\frac{dz_2}{dw_2}
\times
\frac{dL}{dz_2}
=
a_1 \times 2(z_2 - y)
$$
 
### Gradient with respect to $w_1$
 
$$
\frac{dL}{dw_1}
=
\frac{dz_1}{dw_1}
\times
\frac{dL}{dz_2}
\times
\frac{dz_2}{da_1}
\times
\frac{da_1}{dz_1}
$$
 
$$
= x \times 2(z_2 - y) \times w_2 \times \mathrm{ReLU}'(z_1)
$$
 
If $z_1 < 0$, then $\mathrm{ReLU}'(z_1) = 0$ and the whole gradient is 0. No learning signal reaches $w_1$ in that case (a dead neuron).
 
## Input values
 
Values: $x = 2,\; w_1 = 0.5,\; w_2 = 3,\; y = 1$.
 
Forward pass:
 
$$
z_1 = 0.5 \times 2 = 1,\quad
a_1 = 1,\quad
z_2 = 3 \times 1 = 3,\quad
L = (3 - 1)^2 = 4
$$
 
Gradients from the formulas:
 
$$
\frac{dL}{dw_2} = 1 \times 2(3 - 1) = 4
$$
 
$$
\frac{dL}{dw_1} = 2 \times 2(3 - 1) \times 3 \times 1 = 24
$$
 
### Checking the gradients
 
A central difference nudges a weight up and down by $\varepsilon$ and measures how fast $L$ changes:
 
$$
\frac{dL}{dw} \approx \frac{L(w + \varepsilon) - L(w - \varepsilon)}{2\varepsilon}
$$
 
With $\varepsilon = 0.001$, for $w_1$:
 
$$
\frac{L(0.501) - L(0.499)}{0.002} = \frac{4.024036 - 3.976036}{0.002} = 24.0
$$
 
And for $w_2$:
 
$$
\frac{L(3.001) - L(2.999)}{0.002} = \frac{4.004001 - 3.996001}{0.002} = 4.0
$$
 
Both match the formulas, so the derivation is correct. This is the idea behind the gradient checker in the code.
 
## Vectorising inputs

Calculations and their results are found in the [Jupyter Notebook](../notebooks/matrix.ipynb).

$\odot$ means entry-by-entry multiplication.

## Softmax and cross-entropy
 
### Softmax
 
Softmax is a formula which turns a set of values into a probability distribution.
 
- All values are between 0 and 1.
- They all sum to one.
$$
\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$
 
Exponentials make the values positive
 
### Cross-entropy
 
Cross-entropy compares the predicted distribution $p$ with the true distribution $q$. The true distribution is one-hot, so only the correct class $y$ survives:
 
$$
L = -\sum_k q_k \log p_k = -\log p_y
$$
 
How it behaves:
 
- $p_y = 0.99$ gives $L \approx 0.01$, so it is confident and right.
- $p_y = 0.5$ gives $L \approx 0.69$, so it is unsure.
- $p_y = 0.01$ gives $L \approx 4.6$, so it is confident and wrong.
The loss is near 0 when the model is right, and grows quickly when it is confidently wrong. It is the negative log-likelihood, so minimising it is maximum-likelihood training. It is also the K-class version of the binary cross-entropy used in logistic regression.
 
### Gradient
 
Let $S = \sum_j e^{z_j}$. Then:
 
$$
L = -\log \frac{e^{z_y}}{S} = -z_y + \log S
$$
 
Differentiate each term with respect to a general logit $z_k$:
 
$$
\frac{\partial (-z_y)}{\partial z_k} = -\mathbf{1}[k = y]
\qquad
\frac{\partial \log S}{\partial z_k} = \frac{e^{z_k}}{S} = p_k
$$
 
$$
\frac{\partial L}{\partial z_k} = p_k - \mathbf{1}[k = y]
$$
 
In vector form:
 
$$
dz = p - y_{\text{onehot}}
$$
 
Example with $p = [0.659, 0.242, 0.099]$ and $y = 0$:
 
$$
dz = [0.659 - 1,\; 0.242,\; 0.099] = [-0.341,\; 0.242,\; 0.099]
$$
 
The correct logit is pushed up, the wrong logits are pushed down, and the gradients sum to 0 because the probabilities sum to 1.
 
This replaces $dz_2$ in the backward pass above. The other steps are unchanged.
 
## Batch equations
 
For a batch of $m$ examples: $X$ is $m \times d$, $W_1$ is $d \times h$, $b_1$ is $1 \times h$, $W_2$ is $h \times k$, $b_2$ is $1 \times k$, and $Y$ is the $m \times k$ one-hot labels.
 
Forward pass:
 
$$
Z_1 = X W_1 + b_1
$$
 
$$
A_1 = \mathrm{ReLU}(Z_1)
$$
 
$$
Z_2 = A_1 W_2 + b_2
$$
 
$$
P = \mathrm{softmax}(Z_2)
$$
 
$$
L = -\frac{1}{m} \sum_{i=1}^{m} \log P_{i, y_i}
$$
 
Backward pass:
 
$$
dZ_2 = \frac{P - Y}{m}
$$
 
$$
dW_2 = A_1^\top \, dZ_2, \qquad db_2 = \sum_{\text{rows}} dZ_2
$$
 
$$
dA_1 = dZ_2 \, W_2^\top
$$
 
$$
dZ_1 = dA_1 \odot \mathbf{1}[Z_1 > 0]
$$
 
$$
dW_1 = X^\top \, dZ_1, \qquad db_1 = \sum_{\text{rows}} dZ_1
$$
 
The $\frac{1}{m}$ comes from averaging the loss over the batch. The bias gradients sum over rows because each bias is added to every example. The matrix product also sums over the batch.
 
Parameter update with learning rate $\alpha$:
 
$$
W := W - \alpha \, dW, \qquad b := b - \alpha \, db
$$
 
## Gradient checking
 
For each parameter $\theta$, compare the analytic gradient with the central difference:
 
$$
g_{\text{numeric}} = \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}, \qquad \varepsilon \approx 10^{-5}
$$
 
Report the relative error:
 
$$
\frac{\lVert g_{\text{analytic}} - g_{\text{numeric}} \rVert}{\lVert g_{\text{analytic}} \rVert + \lVert g_{\text{numeric}} \rVert}
$$
 
The target is below $10^{-6}$ in double precision.