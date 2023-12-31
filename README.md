# Abstract
Movies are undoubtedly one of the most popular leisure activities for contemporary people
around the world. Arguably film reviews are crucial for movie-goers as they provide a
comprehensive and objective evaluation of films, helping audiences better understand them
and make informed choices. Film reviews also have significant impact on the commercial value
of movies, mainly in terms of influencing film marketing, audience decisions, box office
performance, and award nominations.
<br>
This paper begins by using a pre-existing sentiment "lexicon" to build a baseline model, which
will use positive and negative words and their sentiment strength as features. This approach
will establish a benchmark that subsequent machine learning (ML) models should exceed. Then
it continues to test various binomial classifiers such as Naïve Bayes, Logistic Regression, and
Linear Support Vector Machines with cross-validation to choose parameters for model
selection. One more approach this paper utilizes is neural network model. By comparing
different methods, this paper aims to build a sentiment analyzer to analyse movie reviews for
positive and negative opinions, and ultimately helping addressing a potential parameter for the
movie investors and saving movie-goers hours of time in automatically aggregating this
analysis.
# Introduction
With countless movies being released worldwide, it can be overwhelming for movie enthusiasts
to choose their favorite movies from the vast number of choices available. This is where IMDb
reviews come in handy. Movie enthusiasts usually check the ratings first, selecting high-rated
movies to watch and enjoy.
The reason why IMDb is chosen for analysis is that it has a unique feature of not only providing
overall ratings and proportions for each entry, but also detailed ratings based on gender and
age. Established in 1990, IMDb is an internationally renowned authoritative movie review
website. IMDb's movie rating mechanism is relatively more professional and mysterious. IMDb
is the world's most popular and authoritative website for movies, television, and celebrity
news, with 250 million monthly visits worldwide. In order to allow users to discuss and
communicate with each other, IMDb has always set up a comments section (message boards),
and its official account also has more than 10 million active fans. In addition, there are some
other relatively niche social networking sites that are also popular overseas, such as Snapchat,
Pinterest, Tumblr, etc. Additionally, individuals may wish to generate their own customized list.
Although IMDb's advanced search function is already useful, allowing users to sort results based
on various criteria, it does not provide age and gender classification. Thus, individuals can crawl
the data and sort it themselves to generate a list that meets their age and gender preferences.
Understanding the polarity of movie reviews are of great business and social value as it
provides solid information to both the movie industry (directors, actors, movie investors, and all
relevant parties) and the audience.
This paper tries several approaches to analyze movie reviews, including both baseline models
and machine learning models. The baseline models included sentiment lexicons by William
Hamilton and his colleagues from Stanford and Sentiment Analysis using SentiWordNet. We
also used machine learning models such as Sklearn's CountVectorizer, Multinomial Naive Bayes,
Logistic Regression, and Linear Support Vector Classifier. Finally, we used Keras as a deep
learning framework.
# Data
In this article, the IMDB Movie Review Dataset is utilized, which comprises of 50,000 reviews
divided equally between training and testing sets, each containing 25,000 reviews. The training
and test files are further divided into 12,500 positive reviews and 12,500 negative reviews.
Negative reviews are those reviews in which the reviewer has given a rating of 1 through 4 stars
to the movie, while positive reviews are those rated between 7 and 10 stars. Reviews that
received a rating of 5 or 6 stars are considered neutral and are not included in the analysis.

![](https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_1.png)

![](https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_2.png)

# Data loading and preliminary inspection
The data for training and testing comes from the polarity dataset 2.0, which is collected by Bo
Pang and colleagues from Cornell Univeristy. Upon downloading the dataset, we create two
subfolders (pos/ and neg/) to contain all the positive and negative reviews respectively.
Furthermore, we display the total number of reviews present in both the positive and negative
dictionaries, along with the contents of the first positive and first negative review in each
dictionary.
To utlize the date better, we first inspect basic characteristics to understand the dataset, this
helps gain a better insights to the following questions: <br>
- How many words are there in average for a movie review? <br>
- Regarding the word "content", what information can be inferred from it? For instance,
does the initial positive review consist of numerous positive words? In other words, if
the origin of the review was unknown and it was not specified to be positive, would it
still be clear that it is a positive review? <br>
- Moving on, let's compute some basic statistics, such as the average length of reviews in
the positive and negative categories, the average number of words per sentence, the
size of the vocabulary (i.e., unique words) in each category, and the diversity of the
sentiment vocabulary. <br>
- What is the average frequency of each word in texts of a specific sentiment. <br>
To perform these calculations, we will need to extract words from the text. Since the texts have
already been tokenized and the words are separated by white spaces, we can simply split the
text into words based on these spaces by using the "tokenize" method.
<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_3.png" alt="Image" width="500" height="70" />

The data suggests that positive reviews are, on average, lengthier than negative reviews.
Positive reviews also tend to have longer sentences and a more extensive and diverse
vocabulary.

<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_4.png" alt="Image" width="300" height="70" />


We also discovered that there are 3615 unique words in the positive dictionary and 23345 unique
words in the negative one. We then lemmatize the tokens and recalculate the statistics and
receive the results:
<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_5.png" alt="Image" width="300" height="140" />

The difference between positive and negative reviews becomes less significant, but positive
reviews still appear to be lengthier and more diverse than negative ones.
We believe that adjectives and adverbs are crucial components of movie reviews as they
convey the writer's opinion and help to describe the movie's characteristics and qualities.
Adjectives are used to express the writer's feelings towards the movie and its various elements,
such as the acting, cinematography, and storyline. Adverbs, on the other hand, provide more
context and detail about how these elements are being described. Therefore, we also
investigate whether there are any variations in the number of distinct adjectives and adverbs
used in reviews with varying polarity. It turns out that, similar to the previous findings, positive
reviews contain a signigificantly lower number of distinct adjectives and higher adverbs
compared to negative reviews. 

<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_6.png" alt="Image" width="300" height="70" />


# Baseline Models
Using sentiment lexicons by William Hamilton and his colleagues from Stanford
Initially, we will implement a basic method using sentiment lexicons. William Hamilton and his
research team at Stanford University compiled lists of words that have been assigned sentiment
scores. These lists were categorized based on various factors such as parts of speech, decades,
and domains, including movie-related words. Additionally, the changes in sentiment scores over
time were also recorded, highlighting how some words have evolved to have a more positive
connotation in recent years, such as 'wicked'. With this in mind, we create a straightforward
"classifier" that assesses the sentiment of a review by considering the combined weight of
various sentimental words present in the text. <br>
We utilize a basic approach based on a threshold system, where the cumulative score
determines whether a review is positive or negative. We employ two methods to achieve this: <br>

1. Each positive word carries a weight of +1, while each negative word has a weight of -1;
2. Positive and negative words are assigned different weights based on their sentiment
scores in the lexicon;

<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_7.png" alt="Image" width="300" height="70" />

As a result, we can conclude that the the first four lists are effective enough for positive movie
reviews but of very low accuracy for negative ones, whreas the last list is effective for negative
movie reviews but not the positive one. However, the purpose is to clearly discern the polarity
without knowing whether the movie review is positive or negative in advance, the method is
considered not good enough to serve this purpose.
# Sentiment Analysis using SentiWordNet
One of the limitations of the previous method is that words can have multiple meanings, each
with varying degrees of polarity or strength of sentiment. <br> 
To address this issue, the NLTK library offers access to WordNet, a lexical database that helps
distinguish between different senses of a word. Additionally, researchers from the Text
Learning Group at the University of Pisa have developed SentiWordNet, which is also available
through NLTK. To begin, we determine the number of distinct senses associated with a given
word. By accessing the online version of WordNet, we also view examples of usage and
definitions for each sense. By comparing the two methods, we can see that SentiWordNet
outperform the sentiment lexicons approach.

<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_8.png" alt="Image" width="300" height="200" />

# Machine Learning Methodologies
## Sklearn's CountVectorizer
Initially, we remove punctuation marks and stop words from the data (other filters can be
added as desired) and prepare it for the machine learning pipeline. Next, we randomly shuffle
the data and allocate 80% for training and the remainder for testing. After that, we utilize
sklearn's CountVectorizer to assess the distribution of words across the texts. Finally, we
randomly select 1000 reviews to observe how the process works. As a result, we discover that
the accuracy is 87%, higher than the previous two approaches.
<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_9.png" alt="Image" width="300" height="140" />

## Naive Bayes, Logistic Regression, and Linear Support Vector Classifier
We will now compare the approach of using sklearn's CountVectorizer, as described earlier,
with three different machine learning classifiers: Multinomial Naive Bayes, Logistic Regression,
and Linear Support Vector Classifier.

<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_10.png" alt="Image" width="600" height="420" />

We use feature extraction to convert the text data into a numerical format to represent each
review as a vector of weighted word frequencies. Then, we use Naive Bayes model on the
training set by providing it with the labeled reviews and their corresponding feature vectors
and test it by providing it with unlabeled reviews and comparing its predictions to the actual
labels. As a result, the accuracy for each model is: Multinomial Naive Bayes (86.3%), Logistic
Regression (86.3%), and Linear Support Vector Classifier (88.0%).

<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_11.png" alt="Image" width="400" height="600" />

# Testing Keras
Since Keras effectively learns patterns and relationships within large datasets of text and allows
for more accurate sentiment analysis, which is particularly important when dealing with
subjective content, we believe it is useful for analyzing movie reviews. Additionally, Keras is a
high-level neural network API that is easy to use and has a large community of users, making it
a practical and accessible tool for data analysis. <br>
Our first step is to preprocess the data by tokenizing the text and converting it into numerical
vectors. Next, pad the sequcne to a fixed length and covert the target data to binary labels. We
define, complie, and train the model and discovered that it reaches the accuracy of 88.03%.

<img src="https://github.com/ishani-tagore/IMDB_Sentiment_Analysis_ML_2023/blob/fe784d8336636a71dcdb9acf2020c7f993e8cbf8/ML_Sentiment_12.png" alt="Image" width="300" height="420" />

# Future Direction
One of the limitations of our model is that we do not take “sequence” into account. For that,
we are talking about the relationship between sentences. Since LSTM (Long Short-Term
Memory) is commonly used for sequence analysis (especially analyzing text data), we believe it
is worth a try for future analysis. LSTM is an advancement of the traditional Recurrent Neural
Network (RNN) structure that stores a hidden state to capture dependencies in a data sequence
(like token contexts in movie reviews). It was created to tackle the vanishing gradient issue in
RNNs, where weights for tokens that are distant from the current token in the sequence are
zeroed out, resulting in their exclusion. <br>
To preserve dependencies over a long sequence of data, the LSTM uses an additional cell state
to store relevant contexts that may still be useful for future steps, even if they aren't recent. At
each step, this memory is updated using a sigmoid activation to forget and add new contexts
based on the current inputs and hidden state. Then, another sigmoid activation is used to
selectively update the hidden state for the next step. The cell shown below depicts a single step
in the LSTM sequence.  <br>

Here is our basic idea into utilizing LSTM: <br>
1. Data preparation: the movie review dataset needs to be preprocessed to remove any
irrelevant information and convert into numerical data;
2. Tokenization: The text data needs to be tokenized into individual words or phrases that
can be fed into the LSTM model. This can be done using various tokenization
techniques, such as word-level or character-level tokenization.
3. Embedding: In order for the LSTM model to understand the meaning of the words, the
tokenized words are converted into a vector representation called word embeddings.
These embeddings are learned through an embedding layer in the LSTM model, which
maps each tokenized word to a vector of real numbers.
4. Model training: The LSTM model is trained on the movie review dataset using a set of
labeled examples. During training, the model learns to predict the sentiment of a movie
review based on its word embeddings.
5. Model evaluation: evaluate it on test data
# Conclusion
In our movie review analysis, we evaluated various models for sentiment analysis. We started with baseline models like sentiment lexicons developed by William Hamilton's team and SentiWordNet, which achieved accuracies of 64.8% and 68.8%, respectively. Moving on to machine learning models, we tested Sklearn's CountVectorizer, Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Classifier, with accuracies ranging from 86.3% to 88.0%. Our highest accuracy of 88.03% was achieved with a neural network model developed using Keras. This indicates that Keras is the most effective model for movie review analysis due to its high accuracy and efficiency. <br>

Machine learning is preferred for analyzing movie reviews because it can handle large datasets efficiently, offering a comprehensive overview and uncovering subtle patterns not easily discernible through manual analysis. Movie reviews serve crucial roles in the film industry. They generate publicity and excitement, attracting more viewers and driving box office success, especially before a film's release. Positive reviews boost a movie's performance, while negative ones can hinder box office revenue and award nominations. <br>

Critics' evaluations impact award nominations and provide exposure and praise for movies, benefiting directors and actors in the long term. For audiences, reviews offer a balanced assessment of a film's strengths and weaknesses, aiding informed decision-making. They help viewers understand a film's style, themes, and foster deeper discussions about complex topics. <br>

For future analysis, we recommend the use of Long Short-Term Memory (LSTM) models, as they excel in processing sequential data and have proven effective in natural language processing tasks like sentiment analysis. LSTM can capture longer-term dependencies and relationships within movie reviews, potentially leading to even more accurate insights and analysis. 




