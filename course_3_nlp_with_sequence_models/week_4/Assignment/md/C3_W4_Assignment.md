# Assignment 4:  Question duplicates

Welcome to the fourth assignment of course 3. In this assignment you will explore Siamese networks applied to natural language processing. You will further explore the fundamentals of Trax and you will be able to implement a more complicated structure using it. By completing this assignment, you will learn how to implement models with different architectures. 

## Important Note on Submission to the AutoGrader

Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:

1. You have not added any _extra_ `print` statement(s) in the assignment.
2. You have not added any _extra_ code cell(s) in the assignment.
3. You have not changed any of the function parameters.
4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
5. You are not changing the assignment code where it is not required, like creating _extra_ variables.

If you do any of the following, you will get something like, `Grader not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/sequence-models-in-nlp/supplement/6ThZO/how-to-refresh-your-workspace).

## Outline

- [Overview](#0)
- [Part 1: Importing the Data](#1)
    - [1.1 Loading in the data](#1.1)
    - [1.2 Converting a question to a tensor](#1.2)
    - [1.3 Understanding the iterator](#1.3)
        - [Exercise 01](#ex01)
- [Part 2: Defining the Siamese model](#2)
    - [2.1 Understanding Siamese Network](#2.1)
        - [Exercise 02](#ex02)
    - [2.2 Hard  Negative Mining](#2.2)
        - [Exercise 03](#ex03)
- [Part 3: Training](#3)
    - [3.1 Training the model](#3.1)
        - [Exercise 04](#ex04)
- [Part 4: Evaluation](#4)
    - [4.1 Evaluating your siamese network](#4.1)
    - [4.2 Classify](#4.2)
        - [Exercise 05](#ex05)
- [Part 5: Testing with your own questions](#5)
    - [Exercise 06](#ex06)
- [On Siamese networks](#6)

<a name='0'></a>
### Overview
In this assignment, concretely you will: 

- Learn about Siamese networks
- Understand how the triplet loss works
- Understand how to evaluate accuracy
- Use cosine similarity between the model's outputted vectors
- Use the data generator to get batches of questions
- Predict using your own model

By now, you are familiar with trax and know how to make use of classes to define your model. We will start this homework by asking you to preprocess the data the same way you did in the previous assignments. After processing the data you will build a classifier that will allow you to identify whether two questions are the same or not. 
<img src = "images/meme.png" style="width:550px;height:300px;"/>


You will process the data first and then pad in a similar way you have done in the previous assignment. Your model will take in the two question embeddings, run them through an LSTM, and then compare the outputs of the two sub networks using cosine similarity. Before taking a deep dive into the model, start by importing the data set.


<a name='1'></a>
# Part 1: Importing the Data
<a name='1.1'></a>
### 1.1 Loading in the data

You will be using the Quora question answer dataset to build a model that could identify similar questions. This is a useful task because you don't want to have several versions of the same question posted. Several times when teaching I end up responding to similar questions on piazza, or on other community forums. This data set has been labeled for you. Run the cell below to import some of the packages you will be using. 


```python
import os
import nltk
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import numpy as np
import pandas as pd
import random as rnd

import w4_unittest

nltk.download('punkt')

# set random seeds
rnd.seed(34)
```

    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!


**Notice that for this assignment Trax's numpy is referred to as `fastnp`, while regular numpy is referred to as `np`.**

You will now load in the data set. We have done some preprocessing for you. If you have taken the deeplearning specialization, this is a slightly different training method than the one you have seen there. If you have not, then don't worry about it, we will explain everything. 


```python
data = pd.read_csv("data/questions.csv")
N=len(data)
print('Number of question pairs: ', N)
data.head()
```

    Number of question pairs:  404351





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>qid1</th>
      <th>qid2</th>
      <th>question1</th>
      <th>question2</th>
      <th>is_duplicate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>What is the step by step guide to invest in sh...</td>
      <td>What is the step by step guide to invest in sh...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>What is the story of Kohinoor (Koh-i-Noor) Dia...</td>
      <td>What would happen if the Indian government sto...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>How can I increase the speed of my internet co...</td>
      <td>How can Internet speed be increased by hacking...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>7</td>
      <td>8</td>
      <td>Why am I mentally very lonely? How can I solve...</td>
      <td>Find the remainder when [math]23^{24}[/math] i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>9</td>
      <td>10</td>
      <td>Which one dissolve in water quikly sugar, salt...</td>
      <td>Which fish would survive in salt water?</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We first split the data into a train and test set. The test set will be used later to evaluate our model.


```python
N_train = 300000
N_test  = 10*1024
data_train = data[:N_train]
data_test  = data[N_train:N_train+N_test]
print("Train set:", len(data_train), "Test set:", len(data_test))
del(data) # remove to free memory
```

    Train set: 300000 Test set: 10240


As explained in the lectures, we select only the question pairs that are duplicate to train the model. <br>
We build two batches as input for the Siamese network and we assume that question $q1_i$ (question $i$ in the first batch) is a duplicate of $q2_i$ (question $i$ in the second batch), but all other questions in the second batch are not duplicates of $q1_i$.  
The test set uses the original pairs of questions and the status describing if the questions are duplicates.


```python
td_index = (data_train['is_duplicate'] == 1).to_numpy()
td_index = [i for i, x in enumerate(td_index) if x] 
print('number of duplicate questions: ', len(td_index))
print('indexes of first ten duplicate questions:', td_index[:10])
```

    number of duplicate questions:  111486
    indexes of first ten duplicate questions: [5, 7, 11, 12, 13, 15, 16, 18, 20, 29]



```python
print(data_train['question1'][5])  #  Example of question duplicates (first one in data)
print(data_train['question2'][5])
print('is_duplicate: ', data_train['is_duplicate'][5])
```

    Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?
    I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?
    is_duplicate:  1



```python
Q1_train_words = np.array(data_train['question1'][td_index])
Q2_train_words = np.array(data_train['question2'][td_index])

Q1_test_words = np.array(data_test['question1'])
Q2_test_words = np.array(data_test['question2'])
y_test  = np.array(data_test['is_duplicate'])
```

Above, you have seen that you only took the duplicated questions for training our model. <br>You did so on purpose, because the data generator will produce batches $([q1_1, q1_2, q1_3, ...]$, $[q2_1, q2_2,q2_3, ...])$  where $q1_i$ and $q2_k$ are duplicate if and only if $i = k$.

<br>Let's print to see what your data looks like.


```python
print('TRAINING QUESTIONS:\n')
print('Question 1: ', Q1_train_words[0])
print('Question 2: ', Q2_train_words[0], '\n')
print('Question 1: ', Q1_train_words[5])
print('Question 2: ', Q2_train_words[5], '\n')

print('TESTING QUESTIONS:\n')
print('Question 1: ', Q1_test_words[0])
print('Question 2: ', Q2_test_words[0], '\n')
print('is_duplicate =', y_test[0], '\n')
```

    TRAINING QUESTIONS:
    
    Question 1:  Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?
    Question 2:  I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me? 
    
    Question 1:  What would a Trump presidency mean for current international master’s students on an F1 visa?
    Question 2:  How will a Trump presidency affect the students presently in US or planning to study in US? 
    
    TESTING QUESTIONS:
    
    Question 1:  How do I prepare for interviews for cse?
    Question 2:  What is the best way to prepare for cse? 
    
    is_duplicate = 0 
    


You will now encode each word of the selected duplicate pairs with an index. <br> Given a question, you can then just encode it as a list of numbers.  

First you tokenize the questions using `nltk.word_tokenize`. <br>
You need a python default dictionary which later, during inference, assigns the values $0$ to all Out Of Vocabulary (OOV) words.<br>
Then you encode each word of the selected duplicate pairs with an index. Given a question, you can then just encode it as a list of numbers. 


```python
#create arrays
Q1_train = np.empty_like(Q1_train_words)
Q2_train = np.empty_like(Q2_train_words)

Q1_test = np.empty_like(Q1_test_words)
Q2_test = np.empty_like(Q2_test_words)
```


```python
# Building the vocabulary with the train set         (this might take a minute)
from collections import defaultdict

vocab = defaultdict(lambda: 0)
vocab['<PAD>'] = 1

for idx in range(len(Q1_train_words)):
    Q1_train[idx] = nltk.word_tokenize(Q1_train_words[idx])
    Q2_train[idx] = nltk.word_tokenize(Q2_train_words[idx])
    q = Q1_train[idx] + Q2_train[idx]
    for word in q:
        if word not in vocab:
            vocab[word] = len(vocab) + 1
print('The length of the vocabulary is: ', len(vocab))
```

    The length of the vocabulary is:  36268



```python
print(vocab['<PAD>'])
print(vocab['Astrology'])
print(vocab['Astronomy'])  #not in vocabulary, returns 0
```

    1
    2
    0



```python
for idx in range(len(Q1_test_words)): 
    Q1_test[idx] = nltk.word_tokenize(Q1_test_words[idx])
    Q2_test[idx] = nltk.word_tokenize(Q2_test_words[idx])
```


```python
print('Train set has reduced to: ', len(Q1_train) ) 
print('Test set length: ', len(Q1_test) ) 
```

    Train set has reduced to:  111486
    Test set length:  10240


<a name='1.2'></a>
### 1.2 Converting a question to a tensor

You will now convert every question to a tensor, or an array of numbers, using your vocabulary built above.


```python
# Converting questions to array of integers
for i in range(len(Q1_train)):
    Q1_train[i] = [vocab[word] for word in Q1_train[i]]
    Q2_train[i] = [vocab[word] for word in Q2_train[i]]

        
for i in range(len(Q1_test)):
    Q1_test[i] = [vocab[word] for word in Q1_test[i]]
    Q2_test[i] = [vocab[word] for word in Q2_test[i]]
```


```python
print('first question in the train set:\n')
print(Q1_train_words[0], '\n') 
print('encoded version:')
print(Q1_train[0],'\n')

print('first question in the test set:\n')
print(Q1_test_words[0], '\n')
print('encoded version:')
print(Q1_test[0]) 
```

    first question in the train set:
    
    Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? 
    
    encoded version:
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] 
    
    first question in the test set:
    
    How do I prepare for interviews for cse? 
    
    encoded version:
    [32, 38, 4, 107, 65, 1015, 65, 11509, 21]


You will now split your train set into a training/validation set so that you can use it to train and evaluate your Siamese model.


```python
# Splitting the data
cut_off = int(len(Q1_train)*.8)
train_Q1, train_Q2 = Q1_train[:cut_off], Q2_train[:cut_off]
val_Q1, val_Q2 = Q1_train[cut_off: ], Q2_train[cut_off:]
print('Number of duplicate questions: ', len(Q1_train))
print("The length of the training set is:  ", len(train_Q1))
print("The length of the validation set is: ", len(val_Q1))
```

    Number of duplicate questions:  111486
    The length of the training set is:   89188
    The length of the validation set is:  22298


<a name='1.3'></a>
### 1.3 Understanding the iterator 

Most of the time in Natural Language Processing, and AI in general we use batches when training our data sets. If you were to use stochastic gradient descent with one example at a time, it will take you forever to build a model. In this example, we show you how you can build a data generator that takes in $Q1$ and $Q2$ and returns a batch of size `batch_size`  in the following format $([q1_1, q1_2, q1_3, ...]$, $[q2_1, q2_2,q2_3, ...])$. The tuple consists of two arrays and each array has `batch_size` questions. Again, $q1_i$ and $q2_i$ are duplicates, but they are not duplicates with any other elements in the batch. 

<br>

The command ```next(data_generator)```returns the next batch. This iterator returns the data in a format that you could directly use in your model when computing the feed-forward of your algorithm. This iterator returns a pair of arrays of questions. 

<a name='ex01'></a>
### Exercise 01

**Instructions:**  
Implement the data generator below. Here are some things you will need. 

- While true loop.
- if `index >= len_Q1`, set the `idx` to $0$.
- The generator should return shuffled batches of data. To achieve this without modifying the actual question lists, a list containing the indexes of the questions is created. This list can be shuffled and used to get random batches everytime the index is reset.
- Append elements of $Q1$ and $Q2$ to `input1` and `input2` respectively.
- if `len(input1) == batch_size`, determine `max_len` as the longest question in `input1` and `input2`. Ceil `max_len` to a power of $2$ (for computation purposes) using the following command:  `max_len = 2**int(np.ceil(np.log2(max_len)))`.
- Pad every question by `vocab['<PAD>']` until you get the length `max_len`.
- Use yield to return `input1, input2`. 
- Don't forget to reset `input1, input2`  to empty arrays at the end (data generator resumes from where it last left).


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: data_generator
def data_generator(Q1, Q2, batch_size, pad=1, shuffle=True):
    """Generator function that yields batches of data

    Args:
        Q1 (list): List of transformed (to tensor) questions.
        Q2 (list): List of transformed (to tensor) questions.
        batch_size (int): Number of elements per batch.
        pad (int, optional): Pad character from the vocab. Defaults to 1.
        shuffle (bool, optional): If the batches should be randomnized or not. Defaults to True.
    Yields:
        tuple: Of the form (input1, input2) with types (numpy.ndarray, numpy.ndarray)
        NOTE: input1: inputs to your model [q1a, q2a, q3a, ...] i.e. (q1a,q1b) are duplicates
              input2: targets to your model [q1b, q2b,q3b, ...] i.e. (q1a,q2i) i!=a are not duplicates
    """
    
    input1 = []
    input2 = []
    idx = 0
    len_q = len(Q1)
    question_indexes = [*range(len_q)]
    
    if shuffle:
        rnd.shuffle(question_indexes)
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    while True:
        if idx >= len_q:
            # if idx is greater than or equal to len_q, set idx accordingly 
            # (Hint: look at the instructions above)
            idx = 0
            # shuffle to get random batches if shuffle is set to True
            if shuffle:
                rnd.shuffle(question_indexes) 
                
        # get questions at the `question_indexes[idx]` position in Q1 and Q2
        q1 = Q1[question_indexes[idx]]
        q2 = Q2[question_indexes[idx]]
        
        # increment idx by 1
        idx += 1
        # append q1
        input1.append(q1)
        # append q2
        input2.append(q2)
        if len(input1) == batch_size:
            # determine max_len as the longest question in input1 & input 2
            # Hint: use the `max` function. 
            # take max of input1 & input2 and then max out of the two of them.
            max_len = max(max([len(q) for q in input1]),max([len(q) for q in input2]))
            # pad to power-of-2 (Hint: look at the instructions above)
            max_len = 2 ** int(np.ceil(np.log2(max_len)))
            b1 = [] 
            b2 = [] 
            for q1, q2 in zip(input1, input2):
                # add [pad] to q1 until it reaches max_len
                q1 = q1 + [pad]*(max_len - len(q1))
                # add [pad] to q2 until it reaches max_len
                q2 = q2 + [pad]*(max_len - len(q2))
                # append q1
                b1.append(q1)
                # append q2
                b2.append(q2)
            # use b1 and b2
            yield np.array(b1), np.array(b2)
    ### END CODE HERE ###
            # reset the batches
            input1, input2 = [], []  # reset the batches
```


```python
batch_size = 2
res1, res2 = next(data_generator(train_Q1, train_Q2, batch_size))
print("First questions  : ",'\n', res1, '\n')
print("Second questions : ",'\n', res2)
```

    First questions  :  
     [[ 149  224  225   65  305 1618   39 5924  253   21    1    1    1    1
         1    1]
     [ 676   33    4 1800 9599 5599  131 5853 1852   28 1226   21    1    1
         1    1]] 
    
    Second questions :  
     [[ 149  224  225   39 5924  253   28  460 5438   21    1    1    1    1
         1    1]
     [ 676   33    4 1800    6 6864 6865   11 6866  131 5853 5854 1852   28
      1226   21]]


**Note**: The following expected output is valid only if you run the above test cell **_once_** (first time). The output will change on each execution.

If you think your implementation is correct and it is not matching the output, make sure to restart the kernel and run all the cells from the top again. 

**Expected Output:**
```CPP
First questions  :  
 [[  30   87   78  134 2132 1981   28   78  594   21    1    1    1    1
     1    1]
 [  30   55   78 3541 1460   28   56  253   21    1    1    1    1    1
     1    1]] 

Second questions :  
 [[  30  156   78  134 2132 9508   21    1    1    1    1    1    1    1
     1    1]
 [  30  156   78 3541 1460  131   56  253   21    1    1    1    1    1
     1    1]]
```
Now that you have your generator, you can just call it and it will return tensors which correspond to your questions in the Quora data set.<br>Now you can go ahead and start building your neural network. 




```python
# Test your function
w4_unittest.test_data_generator(data_generator)
```

    [92m All tests passed


<a name='2'></a>
# Part 2: Defining the Siamese model

<a name='2.1'></a>

### 2.1 Understanding Siamese Network 
A Siamese network is a neural network which uses the same weights while working in tandem on two different input vectors to compute comparable output vectors.The Siamese network you are about to implement looks like this:

<img src = "images/siamese.png" style="width:600px;height:300px;"/>

You get the question embedding, run it through an LSTM layer, normalize $v_1$ and $v_2$, and finally use a triplet loss (explained below) to get the corresponding cosine similarity for each pair of questions. As usual, you will start by importing the data set. The triplet loss makes use of a baseline (anchor) input that is compared to a positive (truthy) input and a negative (falsy) input. The distance from the baseline (anchor) input to the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized. In math equations, you are trying to maximize the following.

$$\mathcal{L}(A, P, N)=\max \left(\|\mathrm{f}(A)-\mathrm{f}(P)\|^{2}-\|\mathrm{f}(A)-\mathrm{f}(N)\|^{2}+\alpha, 0\right)$$

$A$ is the anchor input, for example $q1_1$, $P$ the duplicate input, for example, $q2_1$, and $N$ the negative input (the non duplicate question), for example $q2_2$.<br>
$\alpha$ is a margin; you can think about it as a safety net, or by how much you want to push the duplicates from the non duplicates. 
<br>

<a name='ex02'></a>
### Exercise 02

**Instructions:** Implement the `Siamese` function below. You should be using all the objects explained below. 

To implement this model, you will be using `trax`. Concretely, you will be using the following functions.


- `tl.Serial`: Combinator that applies layers serially (by function composition) allows you set up the overall structure of the feedforward. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.combinators.Serial) / [source code](https://github.com/google/trax/blob/1372b903bb66b0daccee19fd0b1fdf44f659330b/trax/layers/combinators.py#L26)
    - You can pass in the layers as arguments to `Serial`, separated by commas. 
    - For example: `tl.Serial(tl.Embeddings(...), tl.Mean(...), tl.Dense(...), tl.LogSoftmax(...))` 


-  `tl.Embedding`: Maps discrete tokens to vectors. It will have shape (vocabulary length X dimension of output vectors). The dimension of output vectors (also called d_feature) is the number of elements in the word embedding. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Embedding) / [source code](https://github.com/google/trax/blob/1372b903bb66b0daccee19fd0b1fdf44f659330b/trax/layers/core.py#L113)
    - `tl.Embedding(vocab_size, d_feature)`.
    - `vocab_size` is the number of unique words in the given vocabulary.
    - `d_feature` is the number of elements in the word embedding (some choices for a word embedding size range from 150 to 300, for example).


-  `tl.LSTM` The LSTM layer. It leverages another Trax layer called [`LSTMCell`](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.rnn.LSTMCell). The number of units should be specified and should match the number of elements in the word embedding. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.rnn.LSTM) / [source code](https://github.com/google/trax/blob/1372b903bb66b0daccee19fd0b1fdf44f659330b/trax/layers/rnn.py#L87)
    - `tl.LSTM(n_units)` Builds an LSTM layer of n_units.
    
    
- `tl.Mean`: Computes the mean across a desired axis. Mean uses one tensor axis to form groups of values and replaces each group with the mean value of that group. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Mean) / [source code](https://github.com/google/trax/blob/1372b903bb66b0daccee19fd0b1fdf44f659330b/trax/layers/core.py#L276)
    - `tl.Mean(axis=1)` mean over columns.


- `tl.Fn` Layer with no weights that applies the function f, which should be specified using a lambda syntax. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.base.Fn) / [source doce](https://github.com/google/trax/blob/70f5364dcaf6ec11aabbd918e5f5e4b0f5bfb995/trax/layers/base.py#L576)
    - $x$ -> This is used for cosine similarity.
    - `tl.Fn('Normalize', lambda x: normalize(x))` Returns a layer with no weights that applies the function `f`
    
    
- `tl.parallel`: It is a combinator layer (like `Serial`) that applies a list of layers in parallel to its inputs. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.combinators.Parallel) / [source code](https://github.com/google/trax/blob/37aba571a89a8ad86be76a569d0ec4a46bdd8642/trax/layers/combinators.py#L152)



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Siamese
def Siamese(vocab_size=41699, d_model=128, mode='train'):
    """Returns a Siamese model.

    Args:
        vocab_size (int, optional): Length of the vocabulary. Defaults to len(vocab).
        d_model (int, optional): Depth of the model. Defaults to 128.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to 'train'.

    Returns:
        trax.layers.combinators.Parallel: A Siamese model. 
    """

    def normalize(x):  # normalizes the vectors to have L2 norm 1
        return x / fastnp.sqrt(fastnp.sum(x * x, axis=-1, keepdims=True))
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    q_processor = tl.Serial( # Processor will run on Q1 and Q2. 
        tl.Embedding(vocab_size=vocab_size, d_feature=d_model), # Embedding layer
        tl.LSTM(n_units=d_model), # LSTM layer
        tl.Mean(axis=1), # Mean over columns
        tl.Fn('Normalize', lambda x: normalize(x)), # Apply normalize function
    )  # Returns one vector of shape [batch_size, d_model]. 
    
    ### END CODE HERE ###
    
    # Run on Q1 and Q2 in parallel.
    model = tl.Parallel(q_processor, q_processor)
    return model

```

Setup the Siamese network model


```python
# check your model
model = Siamese()
print(model)
```

    Parallel_in2_out2[
      Serial[
        Embedding_41699_128
        LSTM_128
        Mean
        Normalize
      ]
      Serial[
        Embedding_41699_128
        LSTM_128
        Mean
        Normalize
      ]
    ]


**Expected output:**  

```CPP
Parallel_in2_out2[
  Serial[
    Embedding_41699_128
    LSTM_128
    Mean
    Normalize
  ]
  Serial[
    Embedding_41699_128
    LSTM_128
    Mean
    Normalize
  ]
]
```


```python
# Test your function
w4_unittest.test_Siamese(Siamese)
```

    [92m All tests passed


<a name='2.2'></a>

### 2.2 Hard  Negative Mining


You will now implement the `TripletLoss`.<br>
As explained in the lecture, loss is composed of two terms. One term utilizes the mean of all the non duplicates, the second utilizes the *closest negative*. Our loss expression is then:
 
\begin{align}
 \mathcal{Loss_{1}(A,P,N)} &=\max \left( -cos(A,P)  + mean_{neg} +\alpha, 0\right) \\
 \mathcal{Loss_{2}(A,P,N)} &=\max \left( -cos(A,P)  + closest_{neg} +\alpha, 0\right) \\
\mathcal{Loss(A,P,N)} &= mean(Loss_1 + Loss_2) \\
\end{align}


Further, two sets of instructions are provided. The first set provides a brief description of the task. If that set proves insufficient, a more detailed set can be displayed.  

<a name='ex03'></a>
### Exercise 03

**Instructions (Brief):** Here is a list of things you should do: <br>

- As this will be run inside trax, use `fastnp.xyz` when using any `xyz` numpy function
- Use `fastnp.dot` to calculate the similarity matrix $v_1v_2^T$ of dimension `batch_size` x `batch_size`
- Take the score of the duplicates on the diagonal `fastnp.diagonal`
- Use the `trax` functions `fastnp.eye` and `fastnp.maximum` for the identity matrix and the maximum.

<details>  
<summary>
    <font size="3" color="darkgreen"><b>More Detailed Instructions </b></font>
</summary>
We'll describe the algorithm using a detailed example. Below, V1, V2 are the output of the normalization blocks in our model. Here we will use a batch_size of 4 and a d_model of 3. As explained in lecture, the inputs, Q1, Q2 are arranged so that corresponding inputs are duplicates while non-corresponding entries are not. The outputs will have the same pattern.
<img src = "images/C3_W4_triploss1.png" style="width:1021px;height:229px;"/>
This testcase arranges the outputs, v1,v2, to highlight different scenarios. Here, the first two outputs V1[0], V2[0] match exactly - so the model is generating the same vector for Q1[0] and Q2[0] inputs. The second outputs differ, circled in orange, we set, V2[1] is set to match V2[2], simulating a model which is generating very poor results. V1[3] and V2[3] match exactly again while V1[4] and V2[4] are set to be exactly wrong - 180 degrees from each other, circled in blue. 

The first step is to compute the cosine similarity matrix or `score` in the code. As explained in lecture, this is $$V_1 V_2^T$$ This is generated with `fastnp.dot`.
<img src = "images/C3_W4_triploss2.png" style="width:959px;height:236px;"/>
The clever arrangement of inputs creates the data needed for positive *and* negative examples without having to run all pair-wise combinations. Because Q1[n] is a duplicate of only Q2[n], other combinations are explicitly created negative examples or *Hard Negative* examples. The matrix multiplication efficiently produces the cosine similarity of all positive/negative combinations as shown above on the left side of the diagram. 'Positive' are the results of duplicate examples and 'negative' are the results of explicitly created negative examples. The results for our test case are as expected, V1[0]V2[0] match producing '1' while our other 'positive' cases (in green) don't match well, as was arranged. The V2[2] was set to match V1[3] producing a poor match at `score[2,2]` and an undesired 'negative' case of a '1' shown in grey. 

With the similarity matrix (`score`) we can begin to implement the loss equations. First, we can extract $$cos(A,P)$$ by utilizing `fastnp.diagonal`. The goal is to grab all the green entries in the diagram above. This is `positive` in the code.

Next, we will create the *closest_negative*. This is the nonduplicate entry in V2 that is closest (has largest cosine similarity) to an entry in V1. Each row, n, of `score` represents all comparisons of the results of Q1[n] vs Q2[x] within a batch. A specific example in our testcase is row `score[2,:]`. It has the cosine similarity of V1[2] and V2[x]. The *closest_negative*, as was arranged, is V2[2] which has a score of 1. This is the maximum value of the 'negative' entries (blue entries in the diagram).

To implement this, we need to pick the maximum entry on a row of `score`, ignoring the 'positive'/green entries. To avoid selecting the 'positive'/green entries, we can make them larger negative numbers. Multiply `fastnp.eye(batch_size)` with 2.0 and subtract it out of `scores`. The result is `negative_without_positive`. Now we can use `fastnp.max`, row by row (axis=1), to select the maximum which is `closest_negative`.

Next, we'll create *mean_negative*. As the name suggests, this is the mean of all the 'negative'/blue values in `score` on a row by row basis. We can use `fastnp.eye(batch_size)` and a constant, this time to create a mask with zeros on the diagonal. Element-wise multiply this with `score` to get just the 'negative values. This is `negative_zero_on_duplicate` in the code. Compute the mean by using `fastnp.sum` on `negative_zero_on_duplicate` for `axis=1` and divide it by `(batch_size - 1)` . This is `mean_negative`.

Now, we can compute loss using the two equations above and `fastnp.maximum`. This will form `triplet_loss1` and `triplet_loss2`. 

`triple_loss` is the `fastnp.mean` of the sum of the two individual losses.

Once you have this code matching the expected results, you can clip out the section between `### START CODE HERE` and `### END CODE HERE` it out and insert it into TripletLoss below. 


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: TripletLossFn
def TripletLossFn(v1, v2, margin=0.25):
    """Custom Loss function.

    Args:
        v1 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        jax.interpreters.xla.DeviceArray: Triplet Loss.
    """
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    
    # use fastnp to take the dot product of the two batches (don't forget to transpose the second argument)
    scores = fastnp.dot(v1, v2.T) # pairwise cosine sim
    #print('scores', scores)
    # calculate new batch size
    batch_size = len(scores) # @KEEPTHIS
    #print('batch_size', batch_size)
    # use fastnp to grab all postive `diagonal` entries in `scores`
    positive = fastnp.diagonal(scores)  # the positive ones (duplicates)
    # multiply `fastnp.eye(batch_size)` with 2.0 and subtract it out of `scores`
    negative_without_positive = scores - fastnp.eye(batch_size)*2.0
    # take the row by row `max` of `negative_without_positive`. 
    # Hint: negative_without_positive.max(axis = None
    closest_negative = negative_without_positive.max(axis=1)
    #print('closest_negative', closest_negative)
    # subtract `fastnp.eye(batch_size)` out of 1.0 and do element-wise multiplication with `scores`
    negative_zero_on_duplicate = (1 - fastnp.eye(batch_size)) * scores
    # use `fastnp.sum` on `negative_zero_on_duplicate` for `axis= None
    mean_negative = fastnp.sum(negative_zero_on_duplicate, axis=1) / (batch_size - 1)
    # compute `fastnp.maximum` among 0.0 and `A`
    # where A = subtract `positive` from `margin` and add `closest_negative`
    # IMPORTANT: DO NOT create an extra variable 'A'
    triplet_loss1 = fastnp.maximum(0.0, margin - positive + closest_negative)
    # compute `fastnp.maximum` among 0.0 and `B`
    # where B = subtract `positive` from `margin` and add `mean_negative`
    # IMPORTANT: DO NOT create an extra variable 'B'
    triplet_loss2 = fastnp.maximum(0.0, margin - positive + mean_negative)
    # add the two losses together and take the `fastnp.mean` of it
    triplet_loss = fastnp.mean(triplet_loss1 + triplet_loss2)
    
    ### END CODE HERE ###
    
    return triplet_loss
```


```python
v1 = np.array([[ 0.26726124,  0.53452248,  0.80178373],[-0.5178918 , -0.57543534, -0.63297887]])
v2 = np.array([[0.26726124, 0.53452248, 0.80178373],[0.5178918 , 0.57543534, 0.63297887]])
print("Triplet Loss:", TripletLossFn(v1,v2))
```

    Triplet Loss: 0.5


**Expected Output:**
```CPP
Triplet Loss: 0.5
```   


```python
# Test your function
w4_unittest.test_TripletLossFn(TripletLossFn)
```

    [92m All tests passed


To make a layer out of a function with no trainable variables, use `tl.Fn`.


```python
from functools import partial
def TripletLoss(margin=0.25):
    triplet_loss_fn = partial(TripletLossFn, margin=margin)
    return tl.Fn('TripletLoss', triplet_loss_fn)
```

<a name='3'></a>

# Part 3: Training

Now you are going to train your model. As usual, you have to define the cost function and the optimizer. You also have to feed in the built model. Before, going into the training, we will use a special data set up. We will define the inputs using the data generator we built above. The lambda function acts as a seed to remember the last batch that was given. Run the cell below to get the question pairs inputs. 


```python
batch_size = 256
train_generator = data_generator(train_Q1, train_Q2, batch_size, vocab['<PAD>'])
val_generator = data_generator(val_Q1, val_Q2, batch_size, vocab['<PAD>'])
print('train_Q1.shape ', train_Q1.shape)
print('val_Q1.shape   ', val_Q1.shape)
```

    train_Q1.shape  (89188,)
    val_Q1.shape    (22298,)


<a name='3.1'></a>

### 3.1 Training the model

You will now write a function that takes in your model and trains it. To train your model you have to decide how many times you want to iterate over the entire data set; each iteration is defined as an `epoch`. For each epoch, you have to go over all the data, using your training iterator.

<a name='ex04'></a>
### Exercise 04

**Instructions:** Implement the `train_model` below to train the neural network above. Here is a list of things you should do, as already shown in lecture 7: 

- Create `TrainTask` and `EvalTask`
- Create the training loop `trax.supervised.training.Loop`
- Pass in the following depending on the context (train_task or eval_task):
    - `labeled_data=generator`
    - `metrics=[TripletLoss()]`,
    - `loss_layer=TripletLoss()`
    - `optimizer=trax.optimizers.Adam` with learning rate of 0.01
    - `lr_schedule=trax.lr.warmup_and_rsqrt_decay(400, 0.01)`,
    - `output_dir=output_dir`


You will be using your triplet loss function with Adam optimizer. Please read the [trax](https://trax-ml.readthedocs.io/en/latest/trax.optimizers.html?highlight=adam#trax.optimizers.adam.Adam) documentation to get a full understanding. 

This function should return a `training.Loop` object. To read more about this check the [docs](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html?highlight=loop#trax.supervised.training.Loop).


```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: train_model
def train_model(Siamese, TripletLoss
                , train_generator, val_generator, output_dir='model/'):
    """Training the Siamese Model

    Args:
        Siamese (function): Function that returns the Siamese model.
        TripletLoss (function): Function that defines the TripletLoss loss function.
        lr_schedule (function): Trax multifactor schedule function.
        train_generator (generator, optional): Training generator. Defaults to train_generator.
        val_generator (generator, optional): Validation generator. Defaults to val_generator.
        output_dir (str, optional): Path to save model to. Defaults to 'model/'.

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """
    output_dir = os.path.expanduser(output_dir)

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    train_task = training.TrainTask( 
        labeled_data=train_generator,      # Use generator (train)
        loss_layer=TripletLoss(),        # Use triplet loss. Don't forget to instantiate this object
        optimizer=trax.optimizers.Adam(learning_rate=0.01),         # Don't forget to add the learning rate parameter
        lr_schedule=trax.lr.warmup_and_rsqrt_decay(400, 0.01) # Use Trax multifactor schedule function
    )

    eval_task = training.EvalTask(
        labeled_data=val_generator,      # Use generator (val)
        metrics=[TripletLoss()],         # Use triplet loss. Don't forget to instantiate this object
    )
    
    ### END CODE HERE ###

    training_loop = training.Loop(Siamese(),
                                  train_task,
                                  eval_tasks=[eval_task],
                                  output_dir=output_dir)

    return training_loop
```


```python
train_steps = 5
training_loop = train_model(Siamese, TripletLoss, train_generator, val_generator)
training_loop.run(train_steps)
```

    Did not manage to match shapes in model for all checkpoint weights.
      Not inserted tensor of shape (41699,)
      Tensor in that place has shape: (41699, 128)
      Not inserted tensor of shape (128,)
      Tensor in that place has shape: (41699, 128)
      Not inserted tensor of shape (256,)
      Tensor in that place has shape: (256, 512)
      Not inserted tensor of shape (512,)
      Tensor in that place has shape: (256, 512)
      Not inserted tensor of shape (512,)
      Tensor in that place has shape: (512,)



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-109-38756ecf7d23> in <module>
          1 train_steps = 5
    ----> 2 training_loop = train_model(Siamese, TripletLoss, train_generator, val_generator)
          3 training_loop.run(train_steps)


    <ipython-input-108-9ee19ed3e04a> in train_model(Siamese, TripletLoss, train_generator, val_generator, output_dir)
         37                                   train_task,
         38                                   eval_tasks=[eval_task],
    ---> 39                                   output_dir=output_dir)
         40 
         41     return training_loop


    /opt/conda/lib/python3.7/site-packages/trax/supervised/training.py in __init__(self, model, tasks, eval_model, eval_tasks, output_dir, checkpoint_at, checkpoint_low_metric, checkpoint_high_metric, permanent_checkpoint_at, eval_at, which_task, n_devices, random_seed, loss_chunk_size, use_memory_efficient_trainer, adasum, callbacks)
        292 
        293     # Load checkpoint if it exists.
    --> 294     self.load_checkpoint()
        295 
        296     # Prepare eval components.


    /opt/conda/lib/python3.7/site-packages/trax/supervised/training.py in load_checkpoint(self, directory, filename)
        941       matched_flat_slots = _match_by_shape(
        942           self._to_bits(_flatten_and_remove_empty(trainer.slots)),
    --> 943           _flatten_and_remove_empty(slots))
        944       matched_slots, _ = fastmath.tree_unflatten(
        945           self._from_bits(matched_flat_slots),


    /opt/conda/lib/python3.7/site-packages/trax/supervised/training.py in _match_by_shape(full, partial)
       1380       model_weight_shape = str(full[i + partial_idx].shape)
       1381       _log('  Tensor in that place has shape: %s' % model_weight_shape)
    -> 1382     raise IndexError
       1383   return res
       1384 


    IndexError: 


The model was only trained for 5 steps due to the constraints of this environment. For the rest of the assignment you will be using a pretrained model but now you should understand how the training can be done using Trax.


```python
# Test your function
w4_unittest.test_train_model(train_model, Siamese, TripletLoss, data_generator)
```

    [92m All tests passed


<a name='4'></a>

# Part 4:  Evaluation  

<a name='4.1'></a>

### 4.1 Evaluating your siamese network

In this section you will learn how to evaluate a Siamese network. You will first start by loading a pretrained model and then you will use it to predict. 


```python
# Loading in the saved model
model = Siamese()
model.init_from_file('model.pkl.gz')
```




    (((array([[-0.1548367 ,  0.5842879 , -0.491473  , ...,  0.20171408,
               -1.1991485 ,  0.24797706],
              [ 1.3291764 , -0.631177  , -0.8041899 , ..., -0.11724447,
                1.1201009 , -1.6056383 ],
              [-0.07363441,  0.60029113, -0.53114337, ...,  0.34642273,
                1.2527287 ,  0.24330486],
              ...,
              [-1.3343723 ,  1.8407435 , -0.9178211 , ..., -0.8050336 ,
               -0.52903235, -0.6519946 ],
              [ 0.6470662 ,  1.1115725 , -1.2664155 , ..., -1.0939023 ,
                0.35975415, -0.9333337 ],
              [-1.7249486 , -0.26522425, -0.8502843 , ...,  0.0217114 ,
               -0.9187999 ,  0.57652336]], dtype=float32),
       (((), ((), ())),
        ((array([[-0.090519  , -0.17231764,  0.08899491, ...,  0.03802169,
                   0.04847963,  0.10207134],
                 [ 0.09968922, -0.12788954, -0.02883548, ..., -0.01054264,
                   0.09648962, -0.14594312],
                 [ 0.31670982,  0.2782086 ,  0.03118353, ...,  0.07789616,
                  -0.10716873,  0.27891394],
                 ...,
                 [-0.07981083, -0.1666319 ,  0.22607869, ..., -0.0145308 ,
                  -0.10772758, -0.07736145],
                 [-0.21396244, -0.27984133,  0.09139609, ..., -0.06796359,
                  -0.13749957, -0.02988345],
                 [ 0.09710262, -0.04201529,  0.23600577, ..., -0.00910869,
                  -0.07826541, -0.06900913]], dtype=float32),
          array([ 0.74866724,  0.23604804,  0.61066806,  0.46742544,  0.84660137,
                  0.6759479 ,  0.75458455,  0.63004017,  0.7348204 ,  0.7810847 ,
                  0.58264714,  0.80906653,  0.56512594,  0.46575448,  0.5894925 ,
                  0.52730995,  0.8338528 ,  0.5772276 ,  0.65321517,  0.597102  ,
                  0.60202324,  0.5726475 ,  0.21979097,  0.30167374,  0.7590757 ,
                  0.823802  ,  0.7020812 ,  0.89135915,  0.6727817 ,  0.62519836,
                  0.52391505,  0.5917081 ,  0.87297624,  0.7066292 ,  0.7613032 ,
                  0.7474745 ,  0.83107007,  0.6900485 ,  0.77471846,  0.42949966,
                  0.49603695,  0.8286887 ,  0.8028574 ,  0.8557113 ,  0.73552644,
                  0.7040203 ,  0.6676977 ,  0.8428011 ,  0.85005385,  0.83686626,
                  0.8439175 ,  0.9119055 ,  0.8712652 ,  0.76832306,  0.94966036,
                  0.9400879 ,  0.40259257,  0.8512523 ,  0.70599616,  0.8428879 ,
                  0.26889625,  0.4928807 ,  0.69416577,  0.8385073 ,  0.63600576,
                  0.665324  ,  0.94755393,  0.21964471,  0.9927555 ,  0.57788414,
                  0.4923829 ,  0.7651692 ,  0.8147946 ,  0.42811084,  0.78248423,
                  0.80160993,  0.0615592 ,  0.42542377,  0.9748414 ,  0.5906    ,
                  0.8284891 ,  0.5356175 ,  0.8901237 ,  0.5789534 ,  0.53019756,
                  0.6117212 ,  0.6344657 ,  0.7970178 ,  0.7678922 ,  0.56659377,
                  0.38351488,  0.46815988,  0.92958826,  0.75536066,  0.7945257 ,
                  0.81163955, -0.03411549,  0.6552565 ,  0.89075   ,  0.12296801,
                  0.5060967 ,  0.6040902 ,  0.706429  ,  0.5993115 ,  0.94161844,
                  0.7323816 ,  0.5769668 ,  0.76575243,  0.82241976,  0.5021452 ,
                  0.39745304,  0.5555639 ,  0.8681804 ,  0.35063615,  0.79127926,
                  0.6759815 ,  0.6214066 ,  0.8014635 ,  0.6679272 ,  0.73128015,
                  0.9667262 ,  0.6471471 ,  0.7636121 ,  0.80299324,  0.56265306,
                  0.78288954,  0.7186831 ,  0.86952096,  0.93183666,  0.9428032 ,
                  1.0660955 ,  0.9234194 ,  0.98314553,  0.9036281 ,  0.9164918 ,
                  0.96618134,  0.9356835 ,  0.9602327 ,  0.9254187 ,  0.97522515,
                  1.0308713 ,  0.96419185,  0.90409976,  0.8842241 ,  1.0028604 ,
                  0.8919036 ,  0.90017885,  1.0468265 ,  0.95297223,  0.8282836 ,
                  0.8075595 ,  0.88657457,  0.99655706,  0.8724078 ,  0.9252944 ,
                  1.0361383 ,  0.91071635,  0.8898618 ,  0.8949262 ,  0.9023262 ,
                  0.93774265,  0.99980384,  0.9492172 ,  0.9804592 ,  1.0260873 ,
                  0.9242303 ,  0.94389015,  1.0269797 ,  0.92412454,  0.8666998 ,
                  0.9186517 ,  0.91434723,  0.93527025,  0.87238604,  0.92576706,
                  1.0864905 ,  0.9672682 ,  0.9451934 ,  0.9366875 ,  0.97492445,
                  1.0422153 ,  0.96439254,  1.0562335 ,  0.9816442 ,  0.9188951 ,
                  0.96140105,  0.94802064,  0.971322  ,  0.909987  ,  0.93718976,
                  0.92520225,  0.9487348 ,  0.9173629 ,  0.9022114 ,  0.98291016,
                  0.9203712 ,  1.0528622 ,  0.92637384,  0.8964494 ,  0.87238085,
                  0.8651392 ,  0.92889744,  0.96058226,  1.03564   ,  1.0819371 ,
                  0.9664545 ,  1.003696  ,  0.9388842 ,  0.9729707 ,  1.0196446 ,
                  0.96996737,  0.9818085 ,  0.8986951 ,  0.915359  ,  0.8793746 ,
                  1.0020589 ,  0.97678035,  0.9033994 ,  0.89549446,  0.91473526,
                  0.96713454,  0.9447215 ,  0.94387925,  0.9418829 ,  0.87416893,
                  0.91230744,  0.98161036,  1.037545  ,  0.96248883,  0.92170656,
                  0.9302868 ,  0.9691254 ,  1.0190876 ,  1.0284687 ,  0.9252369 ,
                  0.8912538 ,  0.9601474 ,  0.9675769 ,  0.9491154 ,  0.96084094,
                  0.9740903 ,  0.91221315,  0.936837  ,  0.92450714,  0.9187654 ,
                  0.9460794 ,  0.85430056,  0.96632856,  0.877542  ,  0.8769376 ,
                  0.96478283,  0.93429935,  0.9077798 ,  0.945345  ,  0.8814642 ,
                  1.0224122 ,  1.1819276 ,  1.1743892 ,  1.1259301 ,  1.1365324 ,
                  1.0201721 ,  0.77878535,  1.1089861 ,  1.3017315 ,  1.0079747 ,
                  0.88814735,  1.1703241 ,  1.0726731 ,  1.2037915 ,  1.2322062 ,
                  1.1280077 ,  1.1659031 ,  0.99659234,  1.0427486 ,  1.0784584 ,
                  1.0045304 ,  1.1760176 ,  1.1747589 ,  1.1270305 ,  1.1475794 ,
                  1.1064138 ,  0.94072586,  1.1092551 ,  1.0405165 ,  1.0915463 ,
                  1.1692449 ,  1.1306821 ,  1.1056664 ,  0.96762   ,  1.044192  ,
                  1.1142045 ,  1.043103  ,  1.1768122 ,  1.095521  ,  0.9751092 ,
                  1.1367201 ,  1.1491212 ,  0.9467336 ,  0.8791624 ,  0.97153306,
                  1.1327726 ,  1.1412548 ,  0.8003143 ,  1.063433  ,  1.0218472 ,
                  0.97348887,  0.8892381 ,  0.9056184 ,  1.0377039 ,  1.0159701 ,
                  0.9115247 ,  0.9698118 ,  1.1991931 ,  1.0183407 ,  0.93539894,
                  1.090577  ,  1.1811533 ,  1.0967308 ,  1.135492  ,  1.049782  ,
                  1.1525736 ,  1.1287878 ,  0.9547346 ,  1.1560799 ,  1.0942348 ,
                  1.1362269 ,  1.1183137 ,  1.1266384 ,  0.9379652 ,  1.0124723 ,
                  1.0144356 ,  1.0998818 ,  1.3319428 ,  1.0870364 ,  0.87949634,
                  1.1151806 ,  1.0183502 ,  0.7299382 ,  0.94768083,  1.1095407 ,
                  1.1219616 ,  1.1669122 ,  1.095584  ,  0.94217265,  0.8934846 ,
                  1.1007344 ,  1.1559246 ,  1.1674374 ,  1.0095923 ,  0.8762718 ,
                  1.0119897 ,  1.0808157 ,  1.3190006 ,  1.1605103 ,  1.045878  ,
                  1.1813673 ,  1.1257927 ,  1.1406978 ,  1.1264644 ,  1.1466924 ,
                  1.1002922 ,  1.170897  ,  1.1333318 ,  1.0016005 ,  1.1115596 ,
                  1.1753446 ,  1.1535839 ,  1.1837677 ,  1.0424278 ,  1.1577648 ,
                  0.94917965,  1.2191473 ,  1.153406  ,  0.85698205,  1.1744432 ,
                  1.1687523 ,  0.83516914,  1.1171178 ,  1.136751  ,  0.93364644,
                  1.1793485 ,  1.0649722 ,  1.0637076 ,  0.948769  ,  0.95589656,
                  0.9806558 ,  0.9719903 ,  0.9753443 ,  0.83935803,  0.99856   ,
                  1.0142294 ,  0.9422684 ,  0.88958335,  0.83948064,  1.0275595 ,
                  0.985004  ,  0.9091298 ,  0.98124814,  0.9453334 ,  0.970391  ,
                  0.98279506,  0.959381  ,  1.0146968 ,  1.0164036 ,  1.0244045 ,
                  0.99479854,  0.8352394 ,  0.9564135 ,  1.013886  ,  0.9127604 ,
                  1.0300385 ,  1.015432  ,  1.0020359 ,  1.0248142 ,  1.0215341 ,
                  0.97478676,  0.988786  ,  0.9202841 ,  0.99523634,  0.94407266,
                  0.88930994,  1.0252285 ,  1.0044427 ,  0.7899437 ,  0.9401787 ,
                  0.9649993 ,  0.906628  ,  0.96424896,  1.0199634 ,  0.96326023,
                  0.9143608 ,  1.0259404 ,  0.8494232 ,  0.96625495,  0.8660369 ,
                  0.8509821 ,  0.9552434 ,  0.79410475,  0.93694836,  0.8247366 ,
                  0.98595744,  0.96510786,  0.8712254 ,  0.88906306,  0.9786764 ,
                  1.0831393 ,  0.9558825 ,  0.94343257,  1.0149219 ,  1.0587987 ,
                  0.88914365,  0.89742774,  0.94791895,  1.0467327 ,  0.8927613 ,
                  0.98992395,  0.9364004 ,  0.9667834 ,  0.92568415,  0.98102164,
                  0.95136875,  0.863204  ,  0.92163223,  1.0005571 ,  0.9337726 ,
                  0.92428803,  0.92654896,  0.95457834,  1.0239131 ,  1.0114415 ,
                  1.0324125 ,  0.9158145 ,  0.8941093 ,  0.99975413,  0.9539995 ,
                  1.014431  ,  0.9227971 ,  0.9062155 ,  0.8609443 ,  0.94716686,
                  0.9145248 ,  1.0938901 ,  0.8660885 ,  1.0017323 ,  0.95409846,
                  1.0142331 ,  0.98524636,  1.0461007 ,  0.8833331 ,  0.9143298 ,
                  1.0509038 ,  0.8950872 ,  1.0015632 ,  1.0264977 ,  1.0189135 ,
                  0.91027725,  0.95289606,  0.99035126,  0.9531608 ,  1.0130571 ,
                  0.9590856 ,  0.92991644,  0.9554187 ,  1.0178174 ,  0.9741912 ,
                  0.98893434,  0.95927787,  0.93184817,  1.0099845 ,  0.94556713,
                  0.90718126,  0.9819065 ], dtype=float32)),),
        ()),
       (),
       ()),
      {'__marker_for_cached_weights_': ()}),
     (((), (((), ((), ())), ((), ()), ()), (), ()),
      {'__marker_for_cached_state_': ()}))



<a name='4.2'></a>
### 4.2 Classify
To determine the accuracy of the model, we will utilize the test set that was configured earlier. While in training we used only positive examples, the test data, Q1_test, Q2_test and y_test, is setup as pairs of questions, some of which are duplicates some are not. 
This routine will run all the test question pairs through the model, compute the cosine simlarity of each pair, threshold it and compare the result to  y_test - the correct response from the data set. The results are accumulated to produce an accuracy.


<a name='ex05'></a>
### Exercise 05

**Instructions**  
 - Loop through the incoming data in `batch_size` chunks
 - Use the data generator to load `q1`, `q2` a batch at a time. **Don't forget to set `shuffle=False`!**
 - copy a `batch_size` chunk of `y` into `y_test`
 - compute `v1`, `v2` using the model
 - for each element of the batch
        - compute the cos similarity of each pair of entries, `v1[j]`,`v2[j]`
        - determine if `d` > threshold
        - increment accuracy if that result matches the expected results (`y_test[j]`)
 - compute the final accuracy and return
 
Due to some limitations of this environment, running classify multiple times may result in the kernel failing. If that happens *Restart Kernal & clear output* and then run from the top. During development, consider using a smaller set of data to reduce the number of calls to model(). 


```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: classify
def classify(test_Q1, test_Q2, y, threshold, model, vocab, data_generator=data_generator, batch_size=64):
    """Function to test the accuracy of the model.

    Args:
        test_Q1 (numpy.ndarray): Array of Q1 questions.
        test_Q2 (numpy.ndarray): Array of Q2 questions.
        y (numpy.ndarray): Array of actual target.
        threshold (float): Desired threshold.
        model (trax.layers.combinators.Parallel): The Siamese model.
        vocab (collections.defaultdict): The vocabulary used.
        data_generator (function): Data generator function. Defaults to data_generator.
        batch_size (int, optional): Size of the batches. Defaults to 64.

    Returns:
        float: Accuracy of the model.
    """    
    
    
    accuracy = 0
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    for i in range(0, len(test_Q1), batch_size):
        # Call the data generator (built in Ex 01) with shuffle= None
        # use batch size chuncks of questions as Q1 & Q2 arguments of the data generator. e.g x[i:i + batch_size]
        # Hint: use `vocab['<PAD>']` for the `pad` argument of the data generator
        q1, q2 = next(data_generator(test_Q1[i: i + batch_size], test_Q2[i: i + batch_size], batch_size, pad=vocab['<PAD>'], shuffle=False))
        # use batch size chuncks of actual output targets (same syntax as example above)
        y_test = y[i: i + batch_size]
        # Call the model    
        v1, v2 = model((q1, q2))

        for j in range(batch_size):
            # take dot product to compute cos similarity of each pair of entries, v1[j], v2[j]
            # don't forget to transpose the second argument
            d = fastnp.dot(v1[j], v2[j].T)
            # is d greater than the threshold?
            res = d > threshold
            # increment accurancy if y_test is equal `res`
            accuracy += y_test[j] == res
    # compute accuracy using accuracy and total length of test questions
    accuracy = accuracy / len(test_Q1)
    ### END CODE HERE ###
    
    return accuracy
```


```python
# this takes around 1 minute
accuracy = classify(Q1_test,Q2_test, y_test, 0.7, model, vocab, batch_size = 512) 
print("Accuracy", accuracy)
```

    Accuracy 0.69091797


**Expected Result**  
Accuracy ~0.69


```python
# Test your function
w4_unittest.test_classify(classify, vocab, data_generator)
```

    [92m All tests passed


<a name='5'></a>

# Part 5: Testing with your own questions

In this section you will test the model with your own questions. You will write a function `predict` which takes two questions as input and returns $1$ or $0$ depending on whether the question pair is a duplicate or not.   

But first, we build a reverse vocabulary that allows to map encoded questions back to words: 

Write a function `predict`that takes in two questions, the model, and the vocabulary and returns whether the questions are duplicates ($1$) or not duplicates ($0$) given a similarity threshold. 

<a name='ex06'></a>
### Exercise 06


**Instructions:** 
- Tokenize your question using `nltk.word_tokenize` 
- Create Q1,Q2 by encoding your questions as a list of numbers using vocab
- pad Q1,Q2 with next(data_generator([Q1], [Q2],1,vocab['<PAD>']))
- use model() to create v1, v2
- compute the cosine similarity (dot product) of v1, v2
- compute res by comparing d to the threshold



```python
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: predict
def predict(question1, question2, threshold, model, vocab, data_generator=data_generator, verbose=False):
    """Function for predicting if two questions are duplicates.

    Args:
        question1 (str): First question.
        question2 (str): Second question.
        threshold (float): Desired threshold.
        model (trax.layers.combinators.Parallel): The Siamese model.
        vocab (collections.defaultdict): The vocabulary used.
        data_generator (function): Data generator function. Defaults to data_generator.
        verbose (bool, optional): If the results should be printed out. Defaults to False.

    Returns:
        bool: True if the questions are duplicates, False otherwise.
    """
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    # use `nltk` word tokenize function to tokenize
    q1 = nltk.word_tokenize(question1)  # tokenize
    q2 = nltk.word_tokenize(question2)  # tokenize
    Q1, Q2 = [], [] # @KEEPTHIS
    for word in q1:  # encode q1
        # increment by checking the 'word' index in `vocab`
        Q1 += [vocab.get(word)]
        # GRADING COMMENT: Also valid to use
        # Q1.extend([vocab[word]])
    for word in q2:  # encode q2
        # increment by checking the 'word' index in `vocab`
        Q2 += [vocab.get(word)]
        # GRADING COMMENT: Also valid to use
        # Q2.extend([vocab[word]])
    
    # GRADING COMMENT: Q1 and Q2 need to be nested inside another list
    # Call the data generator (built in Ex 01) using next()
    # pass [Q1] & [Q2] as Q1 & Q2 arguments of the data generator. Set batch size as 1
    # Hint: use `vocab['<PAD>']` for the `pad` argument of the data generator
    Q1, Q2 = next(data_generator([Q1], [Q2], 1, vocab['<PAD>']))
    # Call the model
    v1, v2 = model((Q1, Q2))
    # take dot product to compute cos similarity of each pair of entries, v1, v2
    # don't forget to transpose the second argument
    d = fastnp.dot(v1, v2.T)
    # is d greater than the threshold?
    res = d > threshold
    
    ### END CODE HERE ###
    
    if(verbose):
        print("Q1  = ", Q1, "\nQ2  = ", Q2)
        print("d   = ", d)
        print("res = ", res)

    return res
```


```python
# Feel free to try with your own questions
question1 = "When will I see you?"
question2 = "When can I see you again?"
# 1 means it is duplicated, 0 otherwise
predict(question1 , question2, 0.7, model, vocab, verbose = True)
```

    Q1  =  [[585  76   4  46  53  21   1   1]] 
    Q2  =  [[ 585   33    4   46   53 7280   21    1]]
    d   =  [[0.8811324]]
    res =  [[ True]]





    DeviceArray([[ True]], dtype=bool)



##### Expected Output
If input is:
```CPP
question1 = "When will I see you?"
question2 = "When can I see you again?"
```

Output is (d may vary a bit):
```CPP
Q1  =  [[585  76   4  46  53  21   1   1]] 
Q2  =  [[ 585   33    4   46   53 7280   21    1]]
d   =  0.88113236
res =  True
True
```


```python
# Feel free to try with your own questions
question1 = "Do they enjoy eating the dessert?"
question2 = "Do they like hiking in the desert?"
# 1 means it is duplicated, 0 otherwise
predict(question1 , question2, 0.7, model, vocab, verbose=True)
```

    Q1  =  [[  443  1145  3159  1169    78 29017    21     1]] 
    Q2  =  [[  443  1145    60 15302    28    78  7431    21]]
    d   =  [[0.47753587]]
    res =  [[False]]





    DeviceArray([[False]], dtype=bool)



##### Expected output

If input is:
```CPP
question1 = "Do they enjoy eating the dessert?"
question2 = "Do they like hiking in the desert?"
```

Output  (d may vary a bit):

```CPP
Q1  =  [[  443  1145  3159  1169    78 29017    21     1]] 
Q2  =  [[  443  1145    60 15302    28    78  7431    21]]
d   =  0.477536
res =  False
False
```

You can see that the Siamese network is capable of catching complicated structures. Concretely it can identify question duplicates although the questions do not have many words in common. 
 


```python
# Test your function
w4_unittest.test_predict(predict, vocab, data_generator)
```

    [92m All tests passed


<a name='6'></a>

###  <span style="color:blue"> On Siamese networks </span>

Siamese networks are important and useful. Many times there are several questions that are already asked in quora, or other platforms and you can use Siamese networks to avoid question duplicates. 

Congratulations, you have now built a powerful system that can recognize question duplicates. In the next course we will use transformers for machine translation, summarization, question answering, and chatbots. 

