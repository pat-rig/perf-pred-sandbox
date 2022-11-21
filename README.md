# Performance Prediction Sandbox
Exploring Methods for Performance Prediction and their Uncertainty Quantification on Images. 

Currently (11/21) contains a MNIST setup including scripts for: 
* Training in ``main.py``
* Retrieval of an embedding in ``embed_train.py`` and ``embed_test.py``, where the new coordinates are layer activations from the trained network
* Notebook with a basic setup for performance predictors in ``neighbour_vs_average.ipynb``

and saved objects:

* ``mnist_cnn.pt`` Trained predictor as Pytorch Model of the class defined in ``main.py`` (2 epochs).
* ``<split>_activations.pt`` DataFrame containing the activations of the last layer (size 128) before the softmax (for each image).
* ``<split>_log_softmax.pt`` DataFrame with softmax activations for each image.
* ``<split>_labels.pt`` DataFrame with labels.
