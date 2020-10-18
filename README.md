# YouTube Spam Comment Filter using Naive Bayes Algorithm
In this notebook, I will be creating a spam filter using multinomial naive bayes algorithm without sci-kit learn, to predict spam comments from Youtube videos. We have data of labelled youtube comments in five seperate files. So we'll train our model with four of them and use the fifth file for prediction and testing for accuracy. The data set used in this notebook can be found [here](https://www.kaggle.com/prashant111/youtube-spam-collection)

## Naive Bayes Algorithm
The Naive Bayes classifier is a simple classifier that classifies based on probabilities of events. It is applied commonly to text classification as although it is a simple algorithm, it performs well in many text classification problems and is faster than other advanced models.
The major drawback of this algorithm is that it assumes that the attributes are independent. This weak point can be solved by performing some statistical analysis before using Naive Bayes to measure the correlation degree among features and then selecting the most uncorrelated ones. Another drawback is the zero probability problem. This can be solved by either adding value one to the frequency of each attribute (Laplace Smoothing) or using Gaussian distribution method.

## Calculation of Probability
We need to find the two probabilities inorder to find out if a comment is spam or not:
* Probability that a comment is spam, given the words in the comment
* Probability that a comment is not spam, given the words in the comment

To find these probabilities, we can use the formula shown below
![alt text](https://github.com/akhilsali/YouTube-Spam-Filter/blob/main/prob1.png)

Inorder to calculate the conditional probabilities of each word (given its a spam or not), we can use the formula shown below:
![alt text](https://github.com/akhilsali/YouTube-Spam-Filter/blob/main/prob2.png)

We will find the probabilities of each class (spam or not), conditional probabilities of each word, number of words in each class using the training data and then use those parameters to predict the class of a comment in the testing data

Note that we will be using Laplace smoothing (alpha=1) for sample correction. We are using a correction to avoid a probability value to become zero just because probability of one of the words in the comment is zero (Since we perform multiplication). Check [this](https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf) blog out for a detailed explaination.
