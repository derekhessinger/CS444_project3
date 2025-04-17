'''cbow.py
The Continuous Bag-of-words (CBOW) neural network
YOUR NAMES HERE
CS444: Deep Learning
Project 3: Word Embeddings
'''
import time
import os
import numpy as np
import tensorflow as tf

import network
from layers import Dense
from cbow_layers import DenseEmbedding

class CBOW(network.DeepNetwork):
    '''Continuous Bag-of-words (CBOW) neural network that learns word embeddings. It consists of the following
    structure:

    Input → DenseEmbedding → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.
    '''
    def __init__(self, C, input_feats_shape, embedding_dim=96, reg=0):
        '''CBOW constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        embedding_dim: int.
            The number of units in the DenseEmbedding layer (H).
        reg: float.
            Regularization strength.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the CBOW network.
        '''
        super().__init__(input_feats_shape, reg)
        self.c = C
        self.embedding_dim = embedding_dim
        self.layers = []
        
        dense_embed = DenseEmbedding('Dense_Embedding', embedding_dim, prev_layer_or_block=None)
        self.layers.append(dense_embed)

        # Output dense layer with softmax activation and He initialization
        self.output_layer = Dense('output_layer', C, activation='softmax', 
                                 wt_init='he', prev_layer_or_block=dense_embed)
        self.layers.append(self.output_layer)

    def __call__(self, x):
        '''Forward pass through the CBOW with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B,).
            Data sample/word INDICES.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.
        '''
        net_act = x
        for cur_layer in self.layers:
            net_act = cur_layer(net_act)
            # print(cur_layer.wts.shape)
            # print(cur_layer.b.shape)
        return net_act

    def fit(self, x, y, batch_size=4096, epochs=32, print_every=1, verbose=True):
        '''Trains CBOW on pairs of context word indices (samples `x`) and target word indices (labels `y`).

        Parameters:
        -----------
        x: tf.constant. tf.int32. shape=(N,).
            Data samples / context word indices in the vocab.
        y: tf.constant. tf.int32. shape=(N,).
            Labels / target word indices in the vocab.
        batch_size: int.
            Number of samples to include in each mini-batch.
        epochs: int.
            Network should train this many epochs.
        print_every: int.
            How often (in epochs) should the network print progress and the training loss?
        verbose: bool.
            If set to `False`, there should be no print outs during training. Messages indicating start and end of
            training are fine.

        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.

        NOTE: This should be MUCH simpler than your existing training loop :)
        NOTE: You are essentially removing/simplying for current training loop here (e.g. remove val set support, early
        stopping, learning rate decay, etc.)
        '''
        # Set all layers to training mode
        self.set_layer_training_mode(is_training=True)

        N = x.shape[0]
        train_loss_hist = []
        np.random.seed(0)
        

        # Main training loop
        for epoch in range(epochs):
            start_time = time.time()
            epoch_train_loss = 0.0

            # how many replacement-based mini-batches
            num_batches = int(np.ceil(N / batch_size))
            epoch_train_loss = 0.0
            
            # Process mini-batches with replacement
            for _ in range(num_batches):
                batch_indices = np.random.choice(N, size=batch_size, replace=True)
                x_batch = tf.gather(x, batch_indices)
                y_batch = tf.gather(y, batch_indices)

                # single forward/backward step
                loss_value = self.train_step(x_batch, y_batch)
                epoch_train_loss += float(loss_value)

            avg_loss = epoch_train_loss / num_batches
            train_loss_hist.append(avg_loss)

            if verbose and (epoch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} — {elapsed:.2f}s")

        if verbose:
            print(f"Finished training after {epochs} epochs.")
        return train_loss_hist

    def get_word_embedding(self, wordind):
        '''Given the word index `wordind` retrieve and return the corresponding embedding vector.'''
        
        x = np.array([wordind,])
        embed = self.layers[0]
        net_act = embed(x)
        return net_act
        pass

    def get_all_embeddings(self):
        '''Retrieve and return the embedding vectors for ALL words in the vocab.'''
        return self.layers[0].wts + self.layers[0].b

    def save_embeddings(self, path='export', filename='embeddings.npz'):
        '''Saves the embeddings to disk.

        This function is provided to you. You should not need to modify it.

        Parameters:
        -----------
        path: str.
            Folder path where the embeddings should be saved.
        filename: str.
            Name of the file to which the embeddings should be saved. Should have a .npz file extension.
        '''
        full_path = os.path.join(path, filename)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        wts = self.get_all_embeddings()
        np.savez_compressed(full_path, embeddings=wts)
