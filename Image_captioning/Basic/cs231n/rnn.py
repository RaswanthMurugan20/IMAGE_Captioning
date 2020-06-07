from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=4096, wordvec_dim=128,
                 hidden_dim=4096, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)
        
        # Intilianize output for the soft attention model weights
        self.params['W_v'] =  np.random.randn(1,hidden_dim)
        self.params['W_v'] /= np.sqrt(1)
        
        self.params['W_g'] =  np.random.randn(1,hidden_dim)
        self.params['W_g'] /= np.sqrt(1)
        
        self.params['W_h'] =  np.random.randn(1,hidden_dim)
        self.params['W_h'] /= np.sqrt(1) 
        
        self.params['Wp1'] =  np.random.randn(vocab_size, hidden_dim)
        self.params['Wp1'] /= np.sqrt(vocab_size) 
        
        self.params['Wp2'] =  np.random.randn(vocab_size, 1)
        self.params['Wp2'] /= np.sqrt(vocab_size) 
        
        
        
        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        
        # Weight and bias for the soft attention transformation.
        W_v =  self.params['W_v']
        W_g =  self.params['W_g']
        W_h =  self.params['W_h']
        Wp1 =  self.params['Wp1']
        Wp2 =  self.params['Wp2']
        
        hidden_dim = W_vocab.shape[0]
        vocab_size = W_vocab.shape[1]
        # non-linear function
        
        def sigmoid(x):
            pos_mask = (x >= 0)
            neg_mask = (x < 0)
            z = np.zeros_like(x)
            z[pos_mask] = np.exp(-x[pos_mask])
            z[neg_mask] = np.exp(x[neg_mask])
            top = np.ones_like(x)
            top[neg_mask] = z[neg_mask]
            return top / (1 + z)
        

        loss, grads = 0.0, {}
        W_vocab = 0*W_vocab
        grads['W_vocab'] = np.zeros_like(W_vocab)

        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        
        #forward pass
        
        
#         print("{} {}".format(features.shape,W_proj.shape))
        ih = np.dot(features, W_proj) + b_proj
        cacheI = (features, W_proj, b_proj)
        wordvec, cacheV = word_embedding_forward(captions_in, W_embed)  
        
        if self.cell_type == 'rnn':
            h, cacheH = rnn_forward(wordvec, ih, Wx, Wh, b)
            
        elif self.cell_type == 'lstm':
            h, cacheH = lstm_forward(wordvec, ih, Wx, Wh, b)
       
        N, T, H = h.shape
        V = vocab_size
        
        #Experiment 1
        cacheS = W_h, W_v, W_g, features, h, Wp1, Wp2, b_vocab
            
#           Testing methods: - 
#           print("{} {} {}".format(features.shape, W_g.shape, h[:,j,:].shape))            
#           print("{} {} {} {}".format(h[:,j,:].shape,Wp1.T.shape,c[:,j][:,None].shape,Wp2.T.shape))
#           feautes Relu change
            
#               FORWARD PASS

#         z = np.zeros((N,T,H))
#         alpha = np.zeros(z.shape)
#         wordscores1 = np.zeros((N, T, vocab_size))
#         c = np.zeros((N, T))
        
#         for j in range(T):

#             z[:,j,:] = (W_h)*(np.tanh(W_v*features + W_g*h[:,j,:])) 
#             alpha[:,j,:] = sigmoid(z[:,j,:])    
#             c[:,j] = np.sum(alpha[:,j,:]*features, axis = 1)
#             wordscores1[:,j,:] = np.dot(h[:,j,:],Wp1.T) + np.dot(c[:,j][:,None],Wp2.T) + b_vocab

#              BACK PROP

#         dh = np.zeros_like(h)
#         dalpha = np.zeros_like(alpha)
#         dc = np.zeros_like(c)
#         dz = np.zeros_like(z)


#         loss, dwordscores1 = temporal_softmax_loss(wordscores1, captions_out, L, W_proj, b_proj, W_embed, Wx, Wh, b, W_h, W_v, W_g, Wp1,             Wp2, b_vocab, W_vocab, mask)                 
        
#         grads['b_vocab'] = dwordscores1.sum(axis=(0, 1))    
#         grads['Wp1'] = np.dot( (dwordscores1.reshape(N * T, V) ).T, h.reshape(N * T, H))
#         grads['Wp2'] = np.dot( (dwordscores1.reshape(N * T, V) ).T, c.reshape(N * T, 1))
#         dc = np.dot(dwordscores1.reshape(N * T, V), Wp2).reshape(N,T)
#         daplha = np.zeros((N,T,H))
            
#         for i in range(N):
#             for j in range(T):
#                 dalpha[i,j,:] = dc[i,j]*features[i]
            
       
#         dz = dalpha*sigmoid(z)*(1-sigmoid(z))
      
#         grads['W_h'] = np.zeros_like(W_h)
#         grads['W_v'] = np.zeros_like(W_v) 
#         grads['W_g'] = np.zeros_like(W_g)
        
#         for i in range(T):
#             grads['W_h'] += np.sum( dz[:,j,:]*(np.tanh(W_v*features + W_g*h[:,j,:])) ,axis = 0)
#             grads['W_v'] += W_h*np.sum(dz[:,j,:]*(1-np.square(np.tanh(W_v*features + W_g*h[:,j,:])))*features, axis = 0)   
#             grads['W_g'] += W_h*np.sum(dz[:,j,:]*(1-np.square(np.tanh(W_v*features + W_g*h[:,j,:])))*h[:,j,:], axis = 0)
#             dh[:,j,:] =  np.dot(dwordscores1[:,j,:],Wp1) + dz[:,j,:]*(1-np.square(np.tanh(W_v*features + W_g*h[:,j,:])))*W_h*W_g        
             
             
    
       
        
        #End
        
        wordscore, cacheS = temporal_affine_forward(h, W_vocab, b_vocab)
        L = 0
        loss, dwordscore = temporal_softmax_loss(wordscore, captions_out, mask)
        
        
        #Backprop
        
        dh, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dwordscore,cacheS)
        
                                                              
        if self.cell_type == 'rnn':
            dx, dih, grads['Wx'], grads['Wh'], grads['b']  = rnn_backward(dh,cacheH)
        elif self.cell_type == 'lstm':
            dx, dih, grads['Wx'], grads['Wh'], grads['b']  = lstm_backward(dh,cacheH)    
        
                                                                   
        grads['W_embed']  = word_embedding_backward(dx, cacheV)
        grads['W_proj'] = np.dot(features.T,dih)
        grads['b_proj'] = np.sum(dih,axis = 0) 
        
#       regularization terms
#         grads['Wx'] += L*Wx/ N
#         grads['b'] += L*b / N
#         grads['W_proj'] += L*W_proj /N
#         grads['W_vocab'] += L*W_vocab /N
#         grads['b_vocab'] += L*b_vocab/N
#         grads['Wh'] += L*Wh/N
#         grads['W_embed'] += L*W_embed/N
#         grads['W_h'] += L*W_h/N
#         grads['W_v'] += L*W_v/N
#         grads['W_g'] += L*W_g/N
#         grads['Wp1'] += L*Wp1/N
#         grads['Wp2'] += L*Wp2/N
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        W_v =  self.params['W_v']
        W_g =  self.params['W_g']
        W_h =  self.params['W_h']
        Wp1 =  self.params['Wp1']
        Wp2 =  self.params['Wp2']
        
        
        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################
        
        inih = np.dot(features, W_proj) + b_proj
        
        prev_h = inih
        
        prev_c = np.zeros(prev_h.shape)
        
        start = np.ones((features.shape[0],1),dtype = int)
                  
        wordvec, _ = word_embedding_forward(self._start, W_embed)
        
        for i in range(max_length):
            
            if self.cell_type == 'rnn':
                next_h, _ = rnn_step_forward(wordvec, prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, prev_c, _ = lstm_step_forward(wordvec, prev_h, prev_c, Wx, Wh, b)     
            
            wordscore = np.dot(next_h, W_vocab)+b_vocab
           
#             z = (W_h)*(np.tanh(W_v*features + W_g*next_h)) 
#             alpha = sigmoid(z)    
#             c = np.sum(alpha*features, axis = 1)
#             wordscore = np.dot(next_h, Wp1.T) + np.dot(c[:,None],Wp2.T) + b_vocab

                       
            captions[:,i]  = np.argmax(wordscore,axis = 1)
            
                     
            wordvec, _ = word_embedding_forward(captions[:,i], W_embed) 
            
            prev_h = next_h
            
            
            
        
        
        
        
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
