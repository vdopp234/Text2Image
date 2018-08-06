# Text2Image

This project uses Generative Adverserial Networks (GAN's) to generate a 256x256 image given a text description. I chose to use a GAN over other generative models, such as Pixel RNN, Gated Pixel CNN, Variational Autoencoders, etc. because GAN's have been shown to generate more clear images, although they are much more difficult to train. Both the generator and the discriminator in GAN's are trained using the gradients of the discriminator, so were the discriminator's loss to reach a maximum/minimum quickly, training cannot take place properly. Instead, we want to reach the Nash Equilibrium, and in order to do this I adopted several extra measures, including using the LeakyReLU activation function and noisy labels for the loss function. 

I used the Tensorflow, Keras, and Numpy libraries for Python to implement my model, and I trained it using the Caltech/UCSD Birds Dataset. My model can be generalized to generate images other than birds by instead using the MS COCO dataset, a larger and more extensive dataset.  

Note: This repository does NOT include the dataset used to train the model, as it took an extraordinarily long time to push the data. Instead, I will leave a link to the dataset below. 

Dataset: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
