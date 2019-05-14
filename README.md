# Generate new text content for Harry Potter with LSTM :smile:
https://medium.com/@patrickhk/generate-new-text-content-for-harry-potter-by-lstm-583066b47c2b

I will demonstrate how to create a text generation model in LSTM with Keras in vocabulary level.<br/>

Let’s begin with some checkpoints:<br/>
1. We need to prepare a clean corpus to train the model, what type of data cleansing is needed? turn to lower case? remove punctuation? remove stop words? it depends on your purpose and expected result.
2. Corpus is simply a text file contains many vocabs/characters, how do you create x_input and y_input?
3. Model doesn’t read raw text, it reads tensor/array/vector. How to convert raw text into tensor?
4. How to map the logits into class label and convert into vocab/character that we human can understand?
5. How to introduce randomness to “create” text with more variations instead of the “best fit” one?

## Let’s start with the LSTM text generation model in vocabulary level
## Data:
Like my previous NLP practice, I use the Harry Potter and the Sorcerer’s Stone.txt as the corpus. I first tokenize the whole corpus str because easier to do data cleansing by list comprehension. I remove punctuation and turn them into lower case. The cleaned corpus have 77589 tokens and 6030 unique token. To achieve better result, we can study the unique token list and tailor some custom filtering/cleansing.<br/>
![p1](https://cdn-images-1.medium.com/max/800/1*9rg0i4zqx30fRCr0zXh9tg.png)
## Encode training input for model:
Decide how many vocab you plan to use to predict the next vocab. People call this parameter as maxlen/seq_length depend on what they like. I use maxlen=50 = use 50 vocabs to predict the 51th vocab. Therefore I turn my corpus into a huge list of sequence, each with 51 vocabs. Total have 77538 sequences.<br/>
![p2](https://cdn-images-1.medium.com/max/800/1*216yssVnJS07eP09SUN9Eg.png)<br/>
Then we pass it into tokenizer to create mapping dictionary and convert text into integer sequence value. We can use the keras.preprocessing.text.Tokenizer and keras.preprocessing.sequence.pad_sequences to help us.<br/>
![p3](https://cdn-images-1.medium.com/max/800/1*Jqe7sZ7qXaJXzP-PggSHiQ.png)<br/>
For each sequence, select the first 50 values as x_train, the last value with OH as y_train and our data set is done.<br/>

## Build and compile the model:
If you have GPU, I recommend using CuDNNLSTM which is 3–4 times faster than LSTM.<br/>
![p4](https://cdn-images-1.medium.com/max/800/1*VREQYSoztw-zosxNTd9t3w.png)<br/>

## Training:
I use Adam(my favourite) with learning rate from 0.001 to 0.0001 for around 270 epoch. Initially I thought overfitting is not a significant problem because at the worsen case the model is trained to be descriptive model but I will introduce randomness when making prediction. (I was wrong, relying on the randomness can output different content but semantic meaning maybe not affected)<br/>
![p5](https://cdn-images-1.medium.com/max/800/1*2nP42FLgjMa3m1_wR8cDWw.png)<br/>
![p6](https://cdn-images-1.medium.com/max/800/1*5Mc8zqldTqkFmq3dBdC6Bg.png)<br/>
## Introduce randomness:
I learn this trick from the book written by Francois Chollet. What he does is to reweight the probability distribution with a factor called temperature.<br/>
```def reweight_distribution(orginal_distribution,temperature):
    distribution=np.log(original_distribution)/temperature
    distribution=np.exp(distribution)
    return distribution/np.sum(distribution)
```    
Higher temperature will make the distribution more random and the model will predict more random vocabs, which are our creation.

## Inference result:
We have to first generate a 50 vocab long sentence called seed text, then our model will use seed text to predict the next vocab, then we update the seed text with our newly generated vocab to predict the next vocab. Repeat this process to generate new text content.<br/>
![p7](https://cdn-images-1.medium.com/max/800/1*uxMHXrJ_DHLh0JwGCxkKGA.png)<br/>

The sentences look nice right? Unfortunately I find it is heavily overfitted after checking the raw text. Let’s apply randomness with temperature =0.9 and 5.<br/>
![p8](https://cdn-images-1.medium.com/max/800/1*DxhcdFpqRWEFJMBn3dHcIg.png)<br/>
![p9](https://cdn-images-1.medium.com/max/800/1*RhApHv9KnVFEIANHIRmrMg.png)<br/>
The output content has changed! The amount of variation depends on the temperature value.

Let’s try with custom input sentence. Remember we have to first encode the text into sequence array for the model.<br/>
![p10](https://cdn-images-1.medium.com/max/800/1*xW3s_0S1enIiR21YU-WfMA.png)<br/>


I try another model with larger corpus and less training epoch. I use all Harry Potter books as corpus and train with the same model structure:<br/>
![p11](https://cdn-images-1.medium.com/max/800/1*8a1LtXHzRhfJPFjVz3kRlQ.png)<br/>
![p12](https://cdn-images-1.medium.com/max/800/1*4HTzpHlepWIxGVnVBX-NRg.png)<br/>
![p13](https://cdn-images-1.medium.com/max/800/1*VL9HiK_SIf7lqfx3ETTYAA.png)<br/>
## And here are some result:
![p14](https://cdn-images-1.medium.com/max/800/1*2x-4ZusPMQ9WEnanzzTM3g.png)<br/>
![p15](https://cdn-images-1.medium.com/max/800/1*ympifPzFLamu5owCFFJMGg.png)<br/>
![p16](https://cdn-images-1.medium.com/max/800/1*KVq1IxWLs94OXpFZkbZA9Q.png)<br/>
## Some points to notice:
1. I cannot apply drop out for the LSTM layers because CuDNNLSTM doesn’t support drop out. Unless I use LSTM/GRU but it will take much longer time to train the model.
2. Accuracy is not a good metrics for measuring text content generation therefore I didn’t include it during training. I should consider using BLEU score for validation.
3. Embedding vector is better than OHE. I should consider first train the embedding layer to learn the semantic representations of vocabs. Then use the trained embedding layer(without gradient update) for mapping to get the embedding vector values for the 51th vocab as the y_input.
4. Should carry out more in depth filtering and cleansing for the corpus, I find some vocabs with wired encoding error in the generated text. For better semantic relationship learning, better keep the punctuation.
5. The second model have 7 times larger corpus than the first model but using the same model architecture. Either my first model overfit or my second model underfit. Should spend more time in the layer structure and hyperparameter tuning.
6. Explore the use of autoencoder

-------------------------------------------------------------------------------------------------------------------------------------
### More about me
[[:pencil:My Medium]](https://medium.com/@patrickhk)<br/>
[[:house_with_garden:My Website]](https://www.fiyeroleung.com/)<br/>
[[:space_invader:	My Github]](https://github.com/fiyero)<br/>
