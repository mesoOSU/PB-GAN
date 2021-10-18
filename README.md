# PB-GAN
Physics-based regularization using generative adversarial networks

This code is adapted from Jason Brownlee's blog post on using Pix2Pix GANs to translate satellite images to google maps images: https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

This code translates a composite (black and white) image to its corresponding stress fields. These stress fields show the stress of the composite image in the 11, 12, and 22 directions. Various regularization techniques are used to incorporate physics to enforce stress equilibrium. The techniques are split up into two categries: regualrization on the discriminator (D), and regularization on the generator (G). The file 'pix2pix_no_physics_included.py' demonstrates how the Pix2Pix GAN translates the stress fiels without physics based regualrization techniques.

# Regualrization on the Discriminator
A sigmoid function is used to calculate the probabilty of the root-mean-square (RMS) score coming from a real or fake divergence field. The divergence fields are calculated from the 11&12 and 12&22 stress fields, which results in two divergence fields for each set of composite-stress fields pairs. The output of the sigmoid probability is then multiplied to the discriminator output. To run this code go to the 'sigmoid_regularization.py' file under the 'D_regularization' folder.

The RMS is also calculated on a patch basis, where the RMS score for a given patch of a divergence field is calculated and the sigmoid function calculates the probability of the RMS score for that patch being real or fake. This gives an array of probabilities that is multiplied by the Discriminator output. See the 'sigmoid_patch_divergences.py' file under the 'D_regularization' folder to run this code.

# Regularization on the Generator
Adding the divergence of the stress fields to the original Pix2Pix loss (which is binary cross entropy loss and L1 loss) is in the file 'Adding_divergence_2_GAN_loss.py' under the folder 'G_regularization". 

The RMS is calculated for each real and fake divergence field pair (RMSfake and RMSreal respectively), which is the plugged into the equation ln(RMSfake/RMSreal). This is done twice, once for the divergence fields coming from the 11&12 stress directions, and again for the 12&22 stress directions. The 2 outputs are then added to the original Pix2Pix loss. To run this code, see the file 'ln(rmsf_rmst)_regualrization.py' under the 'G_regularization' folder.

# Set up Paths
Before running any code, please change the 'graph_path' and 'model_path' to where you would like graphs and models to be saved. 'dataset_path' should be changed to where the training dataset 'high_E_contrast_composite_train.npz' is saved. These paths are defined right after the python packages are imported. 

