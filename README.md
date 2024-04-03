
![Transformer in NYC ([**created from phtofunia](https://photofunia.com/)**)](https://cdn-images-1.medium.com/max/4992/1*99eK1ktrNGPyt4IPowcAgg.png)

## Solving Transformer by Hand: A Step-by-Step Math Example

I have already written a detailed blog on how transformers work using a very small sample of the dataset, which will be my best blog ever because it has elevated my profile and given me the motivation to write more. However, that blog is incomplete as it only covers **20% of the transformer architecture** and contains numerous calculation errors, as pointed out by readers. After a considerable amount of time has passed since that blog, I will be revisiting the topic in this new blog.

I plan to explain the transformer and provide a complete guide with a step-by-step approach to understanding how they work.

## Table of Contents

* [Defining our Dataset](#e273)

* [Finding Vocab Size](#ba43)

* [Encoding](#620c)

* [Calculating Embedding](#ca82)

* [Calculating Positional Embedding](#e6c4)

* [Concatenating Positional and Word Embeddings](#8dc4)

* [Multi Head Attention](#3316)

* [Adding and Normalizing](#4657)

* [Feed Forward Network](#aea8)

* [Adding and Normalizing Again](#7c26)

* [Decoder Part](#7726)

* [Understanding Mask Multi Head Attention](#5906)

* [Calculating the Predicted Word](#9d3f)

* [Important Points](#4b2e)

* [Conclusion](#6872)

## Step 1 — Defining our Dataset

The dataset used for creating ChatGPT is **570 GB.** On the other hand, for our purposes, we will be using a very small dataset to perform numerical calculations visually.

![Our entire dataset containing only three sentences](https://cdn-images-1.medium.com/max/6944/1*PXhg5aLIuJiFDaR6NsVFew.png)

Our entire dataset contains only three sentences, all of which are dialogues taken from a TV show. Although our dataset is cleaned, in real-world scenarios like ChatGPT creation, cleaning a 570 GB dataset requires a significant amount of effort.

## Step 2— Finding Vocab Size

The vocabulary size determines the total number of **unique words** in our dataset. It can be calculated using the below formula, where **N** is the total number of words in our dataset.

![vocab_size formula where N is total number of words](https://cdn-images-1.medium.com/max/7600/1*LYkmmxuX6sRGhPHL1f5Y1g.png)

In order to find N, we need to break our dataset into individual words.

![calculating variable **N**](https://cdn-images-1.medium.com/max/10348/1*mVHFT-0cL-8KnDvLMNpOJA.png)

After obtaining N, we perform a set operation to remove duplicates, and then we can count the unique words to determine the vocabulary size.

![finding vocab size](https://cdn-images-1.medium.com/max/12120/1*ob8UfKnG4pSDwKPluCXZjg.png)

Therefore, the vocabulary size is **23**, as there are **23** unique words in our dataset.

## Step 3 — Encoding

Now, we need to assign a unique number to each unique word.

![encoding our unique words](https://cdn-images-1.medium.com/max/15756/1*2v7umtKxna92ypxPGTJMEw.png)

As we have considered a single token as a single word and assigned a number to it, ChatGPT has considered a portion of a word as a single token using this formula: 1 Token = 0.75 Word

After encoding our entire dataset, it’s time to select our input and start working with the transformer architecture.

## Step 4 — Calculating Embedding

Let’s select a sentence from our corpus that will be processed in our transformer architecture.

![Input sentence for transformer](https://cdn-images-1.medium.com/max/17844/1*ojmh3_rU8Z4PtjGfGMXHfQ.png)

We have selected our input, and we need to find an embedding vector for it. The original paper uses a **512-dimensional embedding vector** for each input word.

![Original Paper uses 512 dimension vector](https://cdn-images-1.medium.com/max/14348/1*K9NvPV-9SDHYiyPLfFOYGA.png)

Since, for our case, we need to work with a smaller dimension of embedding vector to visualize how the calculation is taking place. So, we will be using a dimension of 6 for the embedding vector.

![Embedding vectors of our input](https://cdn-images-1.medium.com/max/15540/1*G7Ggd7nEN856_zbuYZZoLw.png)

These values of the embedding vector are between 0 and 1 and are filled randomly in the beginning. They will later be updated as our transformer starts understanding the meanings among the words.

## Step 5 — Calculating Positional Embedding

Now we need to find positional embeddings for our input. There are two formulas for positional embedding depending on the position of the ith value of that embedding vector for each word.

![Positional Embedding formula](https://cdn-images-1.medium.com/max/15860/1*8r-S_gfexMsy19ppBX12ag.png)

As you do know, our input sentence is **“when you play the game of thrones”** and the starting word is **“when” **with a starting index (POS) value is 0, having a dimension (d) of 6. For i from 0 to 5, we calculate the positional embedding for our first word of the input sentence.

![Positional Embedding for word: **When**](https://cdn-images-1.medium.com/max/8736/1*3kz44sGfStozgw_2aBIWjw.png)

Similarly, we can calculate positional embedding for all the words in our input sentence.

![Calculating Positional Embeddings of our input **(The calculated values are rounded)**](https://cdn-images-1.medium.com/max/9104/1*Zyh9367itZlPnEZLqzv4fQ.png)

## Step 6 — Concatenating Positional and Word Embeddings

After calculating positional embedding, we need to add word embeddings and positional embeddings.

![concatenation step](https://cdn-images-1.medium.com/max/16020/1*-Canm4KHuFzuXePmsC4ZJw.png)

This resultant matrix from combining both matrices (**Word embedding matrix **and **positional embedding matrix**) will be considered as an input to the encoder part.

## Step 7 — Multi Head Attention

A multi-head attention is comprised of many single-head attentions. It is up to us how many single heads we need to combine. For example, LLaMA LLM from Meta has used 32 single heads in the encoder architecture. Below is the illustrated diagram of how a single-head attention looks like.

![Single Head attention in Transformer](https://cdn-images-1.medium.com/max/10164/1*lB_CQAexlaU02D_dKF6XYQ.png)

There are three inputs: **query**, **key**, and **value**. Each of these matrices is obtained by multiplying a different set of weights matrix from the **Transpose **of same matrix that we computed earlier by adding the word embedding and positional embedding matrix.

Let’s say, for computing the query matrix, the set of weights matrix must have the number of rows the same as the number of columns of the transpose matrix, while the columns of the weights matrix can be any; for example, we suppose 4 columns in our weights matrix. The values in the weights matrix are between 0 and 1 randomly, which will later be updated when our transformer starts learning the meaning of these words.

![calculating Query matrix](https://cdn-images-1.medium.com/max/15816/1*E5whYIoC5RF3iHQkX_yTUw.png)

Similarly, we can compute the **key** and **value** matrices using the same procedure, but the values in the weights matrix must be different for both.

![Calculating Key and Value Matrices](https://cdn-images-1.medium.com/max/15552/1*chYDhzxZy0j8WPZH7eaZIw.png)

So, after multiplying matrices, the resultant **query**, **key**, and **values** are obtained:

![Query, Key, Value matrices](https://cdn-images-1.medium.com/max/11720/1*dLl9JhTHacmBDRKFGXNPlg.png)

Now that we have all three matrices, let’s start calculating single-head attention step by step.

![matrix multiplication between Query and Key](https://cdn-images-1.medium.com/max/17280/1*JuaC84jFOHHTke7jwRGUiw.png)

For scaling the resultant matrix, we have to reuse the dimension of our embedding vector, which is 6.

![scaling the resultant matrix with dimension **5**](https://cdn-images-1.medium.com/max/16776/1*Sd8OyZr_nQjT_FqeTM3q3g.png)

The next step of **masking** **is** **optional**, and we won’t be calculating it. Masking is like telling the model to focus only on what’s happened before a certain point and not peek into the future while figuring out the importance of different words in a sentence. It helps the model understand things in a step-by-step manner, without cheating by looking ahead.

So now we will be applying the **softmax** operation on our scaled resultant matrix.

![Applying softmax on resultant matrix](https://cdn-images-1.medium.com/max/16436/1*doJ5FUYSj21RtVXIAnrtmw.png)

Doing the final multiplication step to obtain the resultant matrix from single-head attention.

![calculating the final matrix of single head attention](https://cdn-images-1.medium.com/max/14720/1*gyLQkyFnQ_zlrj84I9IIFQ.png)

We have calculated single-head attention, while multi-head attention comprises many single-head attentions, as I stated earlier. Below is a visual of how it looks like:

![Multi Head attention in Transformer](https://cdn-images-1.medium.com/max/10164/1*YwdVB4szxRw0aUQ0SR784A.png)

Each single-head attention has three inputs: **query**, **key**, and **value**, and each three have a different set of weights. Once all single-head attentions output their resultant matrices, they will all be concatenated, and the final concatenated matrix is once again transformed linearly by multiplying it with a set of weights matrix initialized with random values, which will later get updated when the transformer starts training.

Since, in our case, we are considering a single-head attention, but this is how it looks if we are working with multi-head attention.

![Single Head attention vs Multi Head attention](https://cdn-images-1.medium.com/max/12646/1*J-gusetW_fuJXgAj9X8quQ.png)

In either case, whether it’s single-head or multi-head attention, the resultant matrix needs to be once again transformed linearly by multiplying a set of weights matrix.

![normalizing single head attention matrix](https://cdn-images-1.medium.com/max/18668/1*vMfGtR78ZkBfQ90xzl4WKw.png)

Make sure the linear set of weights matrix number of columns must be equal to the matrix that we computed earlier (**word embedding + positional embedding**) matrix number of columns, because the next step, we will be adding the resultant normalized matrix with (**word embedding + positional embedding**) matrix.

![Output matrix of multi head attention](https://cdn-images-1.medium.com/max/23036/1*nUkMToeFhWuIkZ56h9altQ.png)

As we have computed the resultant matrix for multi-head attention, next, we will be working on adding and normalizing step.

## Step 8 — Adding and Normalizing

Once we obtain the resultant matrix from multi-head attention, we have to add it to our original matrix. Let’s do it first.

![Adding matrices to perform add and norm step](https://cdn-images-1.medium.com/max/17538/1*KNpA-drhNtO1MBeFeGyolA.png)

To normalize the above matrix, we need to compute the mean and standard deviation row-wise for each row.

![calculating meand and std.](https://cdn-images-1.medium.com/max/13080/1*_wrRKKRM7EXVY8eLZLyNwQ.png)

we subtract each value of the matrix by the corresponding row mean and divide it by the corresponding standard deviation.

![normalizing the resultant matrix](https://cdn-images-1.medium.com/max/17392/1*-ixsgxBG9mv4dg4LJwOj9Q.png)

Adding a small value of error prevents the denominator from being zero and avoids making the entire term infinity.

## Step 9 — Feed Forward Network

After normalizing the matrix, it will be processed through a feedforward network. We will be using a very basic network that contains only one linear layer and one ReLU activation function layer. This is how it looks like visually:

![Feed Forward network comparison](https://cdn-images-1.medium.com/max/17392/1*rdYzrzqaH30naDV427k3dA.png)

First, we need to calculate the linear layer by multiplying our last calculated matrix with a random set of weights matrix, which will be updated when the transformer starts learning, and adding the resultant matrix to a bias matrix that also contains random values.

![Calculating Linear Layer](https://cdn-images-1.medium.com/max/12550/1*NDFTk3rHug9dKbmuHgT47g.png)

After calculating the linear layer, we need to pass it through the ReLU layer and use its formula.

![Calculating ReLU Layer](https://cdn-images-1.medium.com/max/9202/1*1v8ozoEMHhNRf1NhFTSblQ.png)

## Step 10 — Adding and Normalizing Again

Once we obtain the resultant matrix from feed forward network, we have to add it to the matrix that is obtained from previous add and norm step, and then normalizing it using the row wise mean and standard deviation.

![Add and Norm after Feed Forward Network](https://cdn-images-1.medium.com/max/17538/1*Wa95_yn9E8I5h0NHiWiipw.png)

The output matrix of this add and norm step will serve as the query and key matrix in one of the multi-head attention mechanisms present in the decoder part, which you can easily understand by tracing outward from the add and norm to the decoder section.

## Step 11 — Decoder Part

The good news is that up until now, we have calculated **Encoder part**, ****all the steps that we have performed, from encoding our dataset to passing our matrix through the feedforward network, are unique. It means we haven’t calculated them before. But from now on, all the upcoming steps that is the remaining architecture of the transformer (**Decoder part**) are going to involve similar kinds of matrix multiplications.

Take a look at our transformer architecture. What we have covered so far and what we have to cover yet:

![Upcoming steps illustration](https://cdn-images-1.medium.com/max/9168/1*na7DVUsXIObNwjqsjvAKJA.png)

We won’t be calculating the entire decoder because most of its portion contains similar calculations to what we have already done in the encoder. Calculating the decoder in detail would only make the blog lengthy due to repetitive steps. Instead, we only need to focus on the calculations of the input and output of the decoder.

When training, there are two inputs to the decoder. One is from the encoder, where the output matrix of the last add and norm layer serves as the **query** and **key** for the second multi-head attention layer in the decoder part. Below is the visualization of it (from [**batool haider](https://www.youtube.com/watch?v=gJ9kaJsE78k&t=596s)**):

![Visualization is from [**Batool Haider](https://www.youtube.com/watch?v=gJ9kaJsE78k&t=596s)**](https://cdn-images-1.medium.com/max/2000/0*1_Zhg960nRqy9MIF.gif)

While the value matrix comes from the decoder after the first **add and norm **step.

The second input to the decoder is the predicted text. If you remember, our input to the encoder is when you play game of thrones so the input to the decoder is the predicted text, which in our case is you win or you die .

But the predicted input text needs to follow a standard wrapping of tokens that make the transformer aware of where to start and where to end.

![input comparison of encoder and decoder](https://cdn-images-1.medium.com/max/16640/1*ad4SiZiYUxMKQu8SBe6a2g.png)

Where <start> and <end> are two new tokens being introduced. Moreover, the decoder takes one token as an input at a time. It means that <start> will be served as an input, and you must be the predicted text for it.

![Decoder input **<start> **word](https://cdn-images-1.medium.com/max/7384/1*NcEJ0iDgKI77inXCybyR9Q.png)

As we already know, these embeddings are filled with random values, which will later be updated during the training process.

Compute rest of the blocks in the same way that we computed earlier in the encoder part.

![Calculating Decoder](https://cdn-images-1.medium.com/max/16320/1*h3CQNwApHPGicChhdoBB_A.png)

Before diving into any further details, we need to understand what masked multi-head attention is, using a simple mathematical example.

## Step 12 — Understanding Mask Multi Head Attention

In a Transformer, the masked multi-head attention is like a spotlight that a model uses to focus on different parts of a sentence. It’s special because it doesn’t let the model cheat by looking at words that come later in the sentence. This helps the model understand and generate sentences step by step, which is important in tasks like talking or translating words into another language.

Suppose we have the following input matrix, where each row represents a position in the sequence, and each column represents a feature:

![inpur matrix for masked multi head attentions](https://cdn-images-1.medium.com/max/9216/1*sTqKpYV2bIzuobu9svkuWw.png)

Now, let’s understand the masked multi-head attention components having two heads:

 1. **Linear Projections (Query, Key, Value): **Assume the linear projections for each head: **Head 1: *Wq*1​,*Wk*1​,*Wv*1**​ and **Head 2: *Wq*2​,*Wk*2​,*Wv*2**​

 2. **Calculate Attention Scores: **For each head, calculate attention scores using the dot product of Query and Key, and apply the mask to prevent attending to future positions.

 3. **Apply Softmax: **Apply the softmax function to obtain attention weights.

 4. **Weighted Summation (Value): **Multiply the attention weights by the Value to get the weighted sum for each head.

 5. **Concatenate and Linear Transformation: **Concatenate the outputs from both heads and apply a linear transformation.

### Let’s do a simplified calculation:

Assuming two conditions

* ***Wq*1​ = *Wk*1 ​= *Wv*1 ​= *Wq*2​ = *Wk*2 ​= *Wv*2​ = *I***, the identity matrix.

* ***Q*=*K*=*V*=Input Matrix**

![Mask Multi Head Attention (**Two Heads**)](https://cdn-images-1.medium.com/max/14080/1*h0Sqeddff4_Xd7I6UU-TrQ.png)

The concatenation step combines the outputs from the two attention heads into a single set of information. Imagine you have two friends who each give you advice on a problem. Concatenating their advice means putting both pieces of advice together so that you have a more complete view of what they suggest. In the context of the transformer model, this step helps capture different aspects of the input data from multiple perspectives, contributing to a richer representation that the model can use for further processing.

## Step 13 — Calculating the Predicted Word

The output matrix of the last add and norm block of the decoder must contain the same number of rows as the input matrix, while the number of columns can be any. Here, we work with 6.

![Add and Norm output of decoder](https://cdn-images-1.medium.com/max/14080/1*5iOM08PiTouFxxm87Layww.png)

The last **add and norm block** resultant matrix of the decoder must be flattened in order to match it with a linear layer to find the predicted probability of each unique word in our dataset (corpus).

![flattened the last add and norm block matrix](https://cdn-images-1.medium.com/max/13056/1*feNuBhdViJo_41qC9vey8Q.png)

This flattened layer will be passed through a linear layer to compute the **logits** (scores) of each unique word in our dataset.

![Calculating Logits](https://cdn-images-1.medium.com/max/18354/1*BK_iGry8sF9XehIGxZtkLw.png)

Once we obtain the logits, we can use the **softmax** function to normalize them and find the word that contains the highest probability.

![Finding the Predicted word](https://cdn-images-1.medium.com/max/17640/1*BGmRI8tL1a6olqIaRkqNbw.png)

So based on our calculations, the predicted word from the decoder is you.

![Final output of decoder](https://cdn-images-1.medium.com/max/12592/1*MRYreFD9RrT3KyQ7R-ww2A.png)

This predicted word you, will be treated as the input word for the decoder, and this process continues until the <end> token is predicted.

## Important Points

 1. The above example is very simple, as it does not involve epochs or any other important parameters that can only be visualized using a programming language like Python.

 2. It has shown the process only until training, while evaluation or testing cannot be visually seen using this matrix approach.

 3. Masked multi-head attentions can be used to prevent the transformer from looking at the future, helping to avoid overfitting your model.

## Conclusion

In this blog, I have shown you a very basic way of how transformers mathematically work using matrix approaches. We have applied positional encoding, softmax, feedforward network, and most importantly, multi-head attention.

In the future, I will be posting more blogs on transformers and LLM as my core focus is on NLP. More importantly, if you want to build your own million-parameter LLM from scratch using Python, I have written a blog on it which has received a lot of appreciation on Medium. You can read it here:
[Building a Million-Parameter LLM from Scratch Using Python](https://levelup.gitconnected.com/building-a-million-parameter-llm-from-scratch-using-python-f612398f06c2)

**Have a great time reading!**
