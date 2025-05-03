# SAM 



SAM~\citep{foretSharpnessAwareMinimizationEfficiently2021} seeks to find parameters that lie in flat regions of the loss landscape by solving:

$$
\begin{equation}
\min_\theta \max_{\|W\|_2 \leq \rho} L(W + \epsilon),
\end{equation}
$$
where $\rho$ controls the perturbation radius.

SAM can be summarized as solving the following bilevel optimization:
$$
\begin{aligned}
\textbf{Outer minimization:} \quad & \min_{W} \mathcal{L}(W + \epsilon^*(W)), \\
\textbf{Inner maximization:} \quad & \epsilon^*(W) = \arg\max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}(W + \epsilon).
\end{aligned}
$$
This approach ensures that the update direction accounts for local sharpness, promoting parameter solutions that generalize better.



## Calcualte Step 

SAM proceeds by two steps per update:

### **Step 1 Compute adversarial perturbation direction**

calculate the  most sharp point $w + \epsilon$ in the perturbation radius  $\rho$  with the input $\mathcal{B}$ ,where $\mathcal{B}$ denotes a batch of training data

**1.1** compute the  origin model's loss and gradient 
$$
\mathcal{L}(W; \mathcal{B}) \\
g=\nabla_{W} \mathcal{L}(W; \mathcal{B})
$$
**1.2** Then, it calculates the normalized perturbation $\epsilon$ in the direction of the gradient:
$$
\begin{equation}
    \epsilon^* = 
    \rho \frac{g}{\|g\|_2}
    
    =\rho \frac{\nabla_{W} \mathcal{L}(W;\mathcal{B})}{\|\nabla_{W} \mathcal{L}(W;\mathcal{B})\|_2}.
    \end{equation}
$$
If adaptive SAM is used, the perturbation is scaled element-wise based on parameter magnitude:
$$
\begin{equation}
\epsilon_i = \rho \cdot \frac{ \| W_i \| \cdot g_i }{ \| \|W \| \cdot g \|_2 },
\end{equation}
$$
where $|\cdot|$ denotes the element-wise norm and scaling.

**1.3** Add noise $\epsilon $ on the origin model 
$$
W_{adv} =  W + \epsilon^*
$$

### **Step 2  Gradient descent at the adversarial point**

**2.1** Keeping $W_{\text{adv}}$ fixed, SAM evaluates the loss again
$$
\mathcal{L}(W+\epsilon^*; \mathcal{B})
$$
**2.2** Computes the gradient at the perturbed point
$$
\begin{equation}
   g_{\text{adv}}= \nabla_{W} \mathcal{L}(W + \epsilon^*; \mathcal{B}).
\end{equation}
$$
**2.3** Finally, SAM restores the original parameters $W$, and applies a standard optimizer step using $g_{\text{adv}}$:
$$
\begin{equation}
W \leftarrow W - \eta \cdot g_{\text{adv}},
\end{equation}
$$
where $\eta$ is the learning rate.



## Varient when apply in DDP 

When using **gradient accumulation** with accumulation steps $N$, the effective update at each global optimization step is based on the cumulative gradient:
$$
g = \sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B}_i),
$$
where each $\mathcal{B}_i$ denotes a mini-batch of data.

In practice, to reduce computational and communication overhead in DDP, **SAM only perturbs the weights during the last accumulation step (step $N$)**, using **only the final mini-batch $\mathcal{B}_N$**.



1 calculate the gradient in the N step 
$$
\mathcal{L}(W; \mathcal{B}) \\
g = \sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B_i})
$$
**1.2** Then, it calculates the normalized perturbation $\epsilon$ in the direction of the gradient:
$$
\begin{equation}
    \epsilon^* = 
    \rho \frac{g}{\|g\|_2}
    
    =\rho \frac{\sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B_i})}{\|\sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B_i})\|_2}.
    \end{equation}
$$
**1.3** Add noise $\epsilon $ on the origin model 
$$
W_{adv} =  W + \epsilon^*
$$

### **Step 2  Gradient descent at the adversarial point**

**2.1** Keeping $W_{\text{adv}}$ fixed, SAM evaluates the loss again,only using the last batch
$$
\mathcal{L}(W+\epsilon^*; \mathcal{B}_N)
$$
**2.2** Computes the gradient at the perturbed point
$$
\begin{equation}
   g_{\text{adv}}= \nabla_{W} \mathcal{L}(W + \epsilon^*; \mathcal{B}_N).
\end{equation}
$$
**2.3** Finally, SAM restores the original parameters $W$, and applies a standard optimizer step using $g_{\text{adv}}$:
$$
\begin{equation}
W \leftarrow W - \eta \cdot g_{\text{adv}},
\end{equation}
$$
where $\eta$ is the learning rate.







## SAM -LoRA

LoRA Mode
$$
W=W_0+\Delta_{\text{LoRA}}W=W_0 + BA
$$
where $W_0 \in \mathbb{R}^{d \times k}$ is the pre-trained and frozen weight, while $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable low-rank matrices with rank $r \ll \min(d, k)$. This reduces trainable parameters and enables efficient fine-tuning.

As a result, the loss function $\mathcal{L}(W; \mathcal{B})$, where $\mathcal{B}$ denotes a batch of training data, is optimized only with respect to the LoRA parameters:
$$
g = \nabla_{BA} \mathcal{L}(W; \mathcal{B}).
$$


### **Step 1: Compute adversarial perturbation direction**

**1.1** At the current parameters $W = W_0 + \Delta{W}$, we compute the gradient:
$$
g = \nabla_{BA} \mathcal{L}(W; \mathcal{B}) = (\nabla_{A} \mathcal{L}(W; \mathcal{B}),\nabla_{B} \mathcal{L}(W; \mathcal{B}))
$$
We then construct a pseudo-gradient in the full parameter space, where frozen parameters have zero gradient, to compute the SAM perturbation:
$$
g = 
\begin{cases}
g_i, & \text{if } \theta_i \text{ is trainable (LoRA)} \\
0, & \text{otherwise}
\end{cases}
$$

$$
g_{B} = \nabla_{B} \mathcal{L}(W; \mathcal{B}),  \\
g_{A} =\nabla_{A} \mathcal{L}(W; \mathcal{B})
$$



**1.2** The perturbation is defined as:
$$
\epsilon = \rho \cdot \frac{\tilde{g}}{\|\tilde{g}\|_2},
$$
the result is 
$$
\epsilon_B =  \rho \cdot \frac{\nabla_{B} \mathcal{L}(W; \mathcal{B})}{\|\nabla_{B} \mathcal{L}(W; \mathcal{B})\|_2}
$$

$$
\epsilon_A =  \rho \cdot \frac{\nabla_{A} \mathcal{L}(W; \mathcal{B})}{\|\nabla_{A} \mathcal{L}(W; \mathcal{B})\|_2}
$$

**1.3** then add noise $\epsilon $ on the origin model 

the result is 
$$
B_{\text{adv}} = (B + \epsilon_{B}) , A_{\text{adv}} = (A + \epsilon_{A})
$$

$$
W_{adv} = W_0+ B_{\text{adv}}A_{\text{adv}} =W_0 +  (B + \epsilon_{B})(A +  \epsilon_{B})
$$


### **Step 2: Evaluate and update**

2.1  Keeping $W_{\text{adv}}$ fixed, SAM evaluates the loss again

We evaluate the loss at the perturbed point:
$$
\mathcal{L}_{\text{adv}} = \mathcal{L}(W_{\text{adv}}; \mathcal{B}) 
= \mathcal{L} ( W +   (B + \epsilon_{B})(A +  \epsilon_{B});\mathcal{B})
$$
**2.2** computes the gradient at the perturbed point

Then, compute the gradient only with respect to $\Delta_{\text{LoRA}}$:
$$
g_{\text{adv}}  
= \nabla_{BA} \mathcal{L}(W_{\text{adv}}; \mathcal{B}) 
= \nabla_{BA} \mathcal{L}( W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B})
$$
so 
$$
g_{(B,\ \text{adv})} = \nabla_{B} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B}) \\

g_{(A,\ \text{adv})} = \nabla_{A} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B})
$$


2.3 Finally, we perform a parameter update:
$$
\Delta_{\text{LoRA}} \leftarrow \Delta_{\text{LoRA}} - \eta \cdot g_{\text{adv}},
$$
where $\eta$ is the learning rate.
$$
B \leftarrow  B - \eta \cdot  \nabla_{B} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B}), \\
A \leftarrow  A - \eta \cdot  \nabla_{A} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B}),
$$
 here , the  noise are only affect the local LoRA paramter rather than the whole model









# RWP 

## Intro 

RWP~\citep{du2023efficient} introduces an expectation-based smoothing loss by perturbing model weights with Gaussian noise.



The smoothed (Bayesian) loss is defined as:
$$
\mathcal{L}_{\text{Bayes}}(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2)} [\mathcal{L}(\theta + \epsilon)]
$$
A mixed loss is then proposed:
$$
\mathcal{L}_m(\theta) = \lambda \cdot \mathcal{L}_{\text{Bayes}}(\theta) + (1 - \lambda) \cdot \mathcal{L}(\theta)
$$
where $\lambda \in [0,1]$ controls the trade-off between robustness and task-specific learning.

the first LBayes(w) provides a smoothed landscape that biases the network towards flat region, while the second L(w) helps recover the necessary local information and better locates the minima that contributes to high performance.



## Calculate Step 

### Step 1: Compute clean gradient

**1.1** Forward and backward on clean weights:
$$
g_0 = \nabla_\theta \mathcal{L}(\theta; \mathcal{B})
$$

------

### Step 2: Compute perturbed gradient

**2.1** Sample noise:
$$
\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|)
$$
**2.2** Evaluate perturbed loss and gradient:
$$
g_1 = \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B})
$$

------

### Step 3: Mixed gradient update

**3.1** Combine gradients:
$$
g = \lambda g_1 + (1 - \lambda) g_0 
 = \lambda \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B}) + (1 - \lambda) \nabla_\theta \mathcal{L}(\theta; \mathcal{B})
$$
**3.2** Update parameters:
$$
\theta \leftarrow \theta - \eta \cdot g
$$

------

## Varient from paper - 

To enhance the optimization process, we utilize two distinct batches of data, namely B1 and B2, for the two gradient steps involved.
$$
g = \lambda g_1 + (1 - \lambda) g_0 
 = \lambda \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B_1}) + (1 - \lambda) \nabla_\theta \mathcal{L}(\theta; \mathcal{B_2})
$$


## Varient when apply in DDP 

calcualate the N step 's gradient, then do one time gradient update 
$$
g = \lambda g_N + (1 - \lambda) \sum_{i=1}^{N-1} g_0 
 = \lambda \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B_N}) +  \sum_{i=1}^{N-1}(1 - \lambda) \nabla_\theta \mathcal{L}(\theta; \mathcal{B_i})
$$






## RWP in LoRA Setting



LoRA Mode
$$
W=W_0+\Delta_{\text{LoRA}}W=W_0 + BA
$$
where $W_0 \in \mathbb{R}^{d \times k}$ is the pre-trained and frozen weight, while $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable low-rank matrices with rank $r \ll \min(d, k)$. This reduces trainable parameters and enables efficient fine-tuning.

As a result, the loss function $\mathcal{L}(W; \mathcal{B})$, where $\mathcal{B}$ denotes a batch of training data, is optimized only with respect to the LoRA parameters:
$$
g = \nabla_{BA} \mathcal{L}(W; \mathcal{B})  \\
g_A = \nabla_{A} \mathcal{L}(W; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) \\
g_B = \nabla_{B} \mathcal{L}(W; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B})
$$




## 1 Add nosiy only on the lora part



### Step 1: Compute clean gradient

**1.1** Forward and backward on clean weights:
$$
g_0 = \nabla_\theta \mathcal{L}(\theta)
$$

$$
g_{(0,A)} = \nabla_{A} \mathcal{L}(W; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) \\
g_{(0,B)} = \nabla_{B} \mathcal{L}(W; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B})
$$


------

### Step 2: Compute perturbed gradient

**2.1** Sample noise:
$$
\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|)
$$
**2.2** Add noise , here only add on Lora part
$$
A_{adv} =  A + \epsilon, B_{adv} = B + \epsilon \\
W_{advLoRA} =  W_0 + (B + \epsilon ) (A + \epsilon)
$$


**2.3** Evaluate perturbed loss and gradient:
$$
g_1 = \nabla_\theta \mathcal{L}(\theta + \epsilon) \\
$$

$$
g_{(1,A)} = \nabla_{A} \mathcal{L}(W_{advLoRA}; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B}) \\
g_{(1,B)} = \nabla_{B} \mathcal{L}(W_{advLoRA}; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B})
$$





------

### Step 3: Mixed gradient update

**3.1** Combine gradients:
$$
g = \lambda g_1 + (1 - \lambda) g_0
$$

$$
\begin{aligned}
g_A 
&= a \cdot  g_{(0,A)} + b \cdot g_{(1,A)}  \\ 
&= a \nabla_{A} \mathcal{L}(W; \mathcal{B})+ b \nabla_{A} \mathcal{L}(W_{advLoRA}; \mathcal{B}) \\
&= a \nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{A} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B})  \\

\end{aligned}
$$


$$
\begin{aligned}
g_B 
&= a \cdot  g_{(0,B)} + b \cdot g_{(1,B)}  \\ 
&= a \nabla_{B} \mathcal{L}(W; \mathcal{B})+ b \nabla_{B} \mathcal{L}(W_{advLoRA}; \mathcal{B}) \\
&= a \nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{B} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B})  \\

\end{aligned}
$$


**3.2** Update parameters:
$$
\theta \leftarrow \theta - \eta \cdot g
$$

$$
A \leftarrow A - \eta \cdot g_A \\
B \leftarrow B - \eta \cdot g_B 
$$







## 1 Add nosiy  the whole  model 



### Step 1: Compute clean gradient

**1.1** Forward and backward on clean weights:
$$
g_0 = \nabla_\theta \mathcal{L}(\theta)
$$

$$
g_{(0,A)} = \nabla_{A} \mathcal{L}(W; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) \\
g_{(0,B)} = \nabla_{B} \mathcal{L}(W; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B})
$$


------

### Step 2: Compute perturbed gradient

**2.1** Sample noise:
$$
\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|)
$$
**2.2** Add noise , try to on the whole model 
$$
A_{adv} =  A + \epsilon, B_{adv} = B + \epsilon \\
W_{advFull} =  (W_0+\epsilon) + (B + \epsilon ) (A + \epsilon) 
\sim W_0+ BA + \epsilon   
$$


**2.3** Evaluate perturbed loss and gradient:
$$
g_1 = \nabla_\theta \mathcal{L}(\theta + \epsilon) \\
$$

$$
g_{(1,A)} = \nabla_{A} \mathcal{L}(W_{advFull}; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B}) \\
g_{(1,B)} = \nabla_{B} \mathcal{L}(W_{advFull}; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B})
$$





------

### Step 3: Mixed gradient update

**3.1** Combine gradients:
$$
g = \lambda g_1 + (1 - \lambda) g_0
$$

$$
\begin{aligned}
g_A 
&= a \cdot  g_{(0,A)} + b \cdot g_{(1,A)}  \\ 
&= a \nabla_{A} \mathcal{L}(W; \mathcal{B})+ b \nabla_{A} \mathcal{L}(W_{advFull}; \mathcal{B}) \\
&= a \nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{A} \mathcal{L}(W_0 +BA + \epsilon; \mathcal{B})  \\

\end{aligned}
$$


$$
\begin{aligned}
g_B 
&= a \cdot  g_{(0,B)} + b \cdot g_{(1,B)}  \\ 
&= a \nabla_{B} \mathcal{L}(W; \mathcal{B})+ b \nabla_{B} \mathcal{L}(W_{advLoRA}; \mathcal{B}) \\
&= a \nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{B} \mathcal{L}(W_0 +BA + \epsilon; \mathcal{B})  \\

\end{aligned}
$$


**3.2** Update parameters:
$$
\theta \leftarrow \theta - \eta \cdot g
$$

$$
A \leftarrow A - \eta \cdot g_A \\
B \leftarrow B - \eta \cdot g_B 
$$









