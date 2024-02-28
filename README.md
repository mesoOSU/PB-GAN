# PBR-GAN
Physics-based regularization (PBR) using generative adversarial networks

Code is adapted from Jason Brownlee's blog post on using Pix2Pix GANs to translate satellite images to google maps images: https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

This code translates a high elastic contrast composite image to its corresponding (normalized) 2D stress fields in the 11, 12, and 22 directions. Various physics-based regularization terms enforcing stress equilibrium are incorporated into either the generator or discriminator loss. This code was was used to study the affect of different learning rates and loss weights on different loss fucnitons, so you will notice that each implementation has different learning rate and loss weight values. The model variation across different training sessions for the same implementation was also studied using this code.

## Baseline Method
The baseline method usign the original Pix2Pix objective, with an L2-regularization instead of L1:
 ```math
 V_{Pix2Pix}=\underbrace{min}_{G}\underbrace{max}_{D} \mathbb{E}_{X\sim P_{data}\ } [logD(X,Y)] + \mathbb{E}_{Z\sim P_{Z}\ } [log(1-D(G(Z,Y),Y))]  + \beta L2(G(Z,Y))
 ```
## Physcis Based Regularization Methods
All regularization methods are enforcing stress equilibrium defined by the divergence of stress, $\nabla\cdot\sigma=0$. For 2D stress fields, there will be two divergence fields in total, defined as
```math
K_1(\sigma)=\frac{\partial\sigma_{11}}{\partial x_1}+\frac{\partial\sigma_{12}}{\partial x_2},\ \ K_2(\sigma)=\frac{\partial\sigma_{12}}{\partial x_1}+\frac{\partial\sigma_{22}}{\partial x_2}
```
### Add Divergence
Add the divergence of generated stress fields like a regularization term to the generator loss. Model is penalized if it deviates from zero divergence.
```math
 V_{Add Div}= V_{Pix2Pix} + \gamma \left(|K_{1}(G(Z,Y))| + |K_{2}(G(Z,Y))|\right)
```
### $Tan^{-1}$
The divergence from the dataset is not exactly zero divergence, but converges to a value very close to zero. The idea here is to get the model to converge to a similar stress equilibrium solution similar to that of the training dataset. The root-mean-square values (RMS) of the divergence field is evaluated for both trarget and generated stress fields and comapared through the difference with a $tan^{-1}$ funciton applied to help with gradient updates.
```math
V_{tan^{-1}}= V_{Pix2Pix} + \gamma \left(|tan^{-1}(RMS(K_{1}(G(Z,Y))))-RMS(K_{1}(X))| + |tan^{-1}(RMS(K_{2}(G(Z,Y))))-RMS(K_{2}(X))|\right)
```
### Sigmoid
Similar to the $tan^{-1}$ method, this method evaluates whether a divergence field was calculated from a set of target or generated stress fields, which is then multiplied to the original discriminator loss. The divergence probability is evaluated through a sigmoid funciton

```math
M_{i} = log_{10}(RMS({K}_{i}(\sigma)))
```
```math
S_{i} = -2 \left(\frac{1}{1+e^{-M_{i}}}-0.5\right)
```
With $M_{i}# being the magnitude of the RMS for a divergence field (with i=1,2, indicating which divergence field) and $S_{i}$ indicating the probability of a divergence field coming from target or generated stress fields. The discriminator loss becomes
```math
D_{sig} = D(\sigma,Y)S_{1}S_{2}
```
and the final objective becomes
```math
V_{Sigmoid}=\underbrace{min}_{G}\underbrace{max}_{D} \mathbb{E}_{X\sim P_{data}\ } [logD_{sig}(X,Y)] 
    +\mathbb{E}_{Z\sim P_{Z}\ } [log(1-D_{sig}(G(Z,Y),Y))] 
    + \beta L2(G(Z,Y))
```
  
# Required Python Packages

Tensorflow=2.6.2  
Keras=2.6.0  
matplotlib=3.6.2  
numpy=1.19.5  

# Datasets and models
The train, test, and valiation are avaialble in the link below. Dataset was generated using CP-FFT. The best performing models from to different training sessions for each method are also available. A separate study with 30 different training sessions for each method was done, and the best median and worst performing models for each method are avaible in the same link.  
https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/lenau_1_buckeyemail_osu_edu/EuVrFbk_eglNj8vIRJF2XwUB57Bc1G5r-FoqqnfJg7HgrQ?e=3zWfsW
