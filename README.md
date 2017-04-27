![Image](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/img/93d5f08a4f82d4c.png)

This is support code for the codelab "[Tensorflow and deep learning - without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist)"

The presentation explaining the underlying concepts is [here](https://goo.gl/pHeXe7) and you will find codelab instructions to follow on its last slide. Do not forget to open the speaker notes in the presentation, a lot of the explanations are there.

The lab takes 2.5 hours and takes you through the design and optimisation of a neural network for recognising handwritten digits, from the simplest possible solution all the way to a recognition accuracy above 99%. It covers dense and convolutional networks, as well as techniques such as learning rate decay and dropout.

Installation instructions [here](INSTALL.txt). The short version is: install Python3, then pip3 install tensorflow and matplotlib.   

---


### 5 layer vs Convolutional network performance comparison

The convolutional networks are made by replacing fully-connected layers with convolutional layers, one by one.
The first convolutional network, 3.01, replaces the first fully-connected layer with 

|   Type of network         | File                             | Accuracy |
|---------------------------|----------------------------------|----------|
|  5 layer fully-connected  | mnist_1.7_lrdecay_momentum.py    | 0.9755   |
|  1 convo, 4 layer FC      | mnist_3.01_convo_1layer.py       | 0.9636   |
|  2 convo, 3 layer FC      | mnist_3.02_convo_1layer.py       |  |
|  1 convo, 4 layer FC      | mnist_3.01_convo_1layer.py       |  |
|
