import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from pprintpp import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


# 1 - Loading the data
df = pd.read_csv('amazon-fine-food-reviews.csv')
# print(df.head())

text_first_row = df['Text'].values[0]
# print(text_first_row)

# print(df.shape)

df = df.head(500)  # reducing dataset size
# print(df.shape)

# 2 - Quick EDA
# ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars',
#                                                   figsize=(10, 5))
# ax.set_xlabel('Review Stars')
# plt.show()

# 3 - Basic NLTK
example = df['Text'][50]
# print(example)

# Tokenizers divide strings into lists of substrings :
tokens = nltk.word_tokenize(example)
# print(tokens[:10])
# Example : ['This', 'oatmeal', 'is', 'not', 'good', '.', 'Its', 'mushy', ',', 'soft']

# pos_tag() assigns a part-of-speech tag to each word indicating its grammatical category
# and function in the sentence :
tagged = nltk.pos_tag(tokens)
# print(tagged[:5])
# Example : [('This', 'DT'), ('oatmeal', 'NN'), ('is', 'VBZ'), ('not', 'RB'), ('good', 'JJ')]

# ne_chunk() identifies and classifies named entities such as :
# PERSON, ORGANIZATION or GPE (Geopolitical Entity)
# the named entity recognized by ne_chunk() is "(ORGANIZATION Quaker Oats)"
entities = nltk.chunk.ne_chunk(tagged)
# print(entities)

# 4 - VADER : Sentiment Scoring
sia = SentimentIntensityAnalyzer()

# Short sentences :
polarity_score_a = sia.polarity_scores('I am so happy!')
# print(polarity_score_a)

polarity_score_b = sia.polarity_scores('This is the worst thing ever.')
# print(polarity_score_b)

# We return to our example :
example_score = sia.polarity_scores(example)
# print(example_score)  # {'neg': 0.22, 'neu': 0.78, 'pos': 0.0, 'compound': -0.5448}

# Running the polarity score on the entire dataset :
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):  # calculating and storing the polarity score
    text = row['Text']                             # for each row in a dictionary
    my_id = row['Id']
    res[my_id] = sia.polarity_scores(text)

# pprint(res)

# converting the res dictionary to a dataframe :
vaders = pd.DataFrame(res).T
# print(vaders)
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
# print(vaders)

# Data and metadata :
# print(vaders.head())

# Plot VADER results :
# ax = sns.barplot(data=vaders, x='Score', y='compound')
# ax.set_title('Compound Score by Amazon Star Review')
# plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])

axs[0].set_title('Positive')
axs[0].set_xlabel('Rating')
axs[0].set_ylabel('Polarity')

axs[1].set_title('Neutral')
axs[1].set_xlabel('Rating')
axs[1].set_ylabel('Polarity')

axs[2].set_title('Negative')
axs[2].set_xlabel('Rating')
axs[2].set_ylabel('Polarity')

# plt.show()

# 5 - RoBERTa : Pre-trained Model
# Initializing the RoBERTa Model :
# a RoBERTa model fine-tuned on Twitter data for sentiment analysis :
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# loading the tokenizer associated with the specified pre-trained model :
tokenizer = AutoTokenizer.from_pretrained(MODEL)
# loading the pre-trained model for sequence classification :
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Running RoBERTa Model :
# encoded_text = tokenizer(example, return_tensors='pt')
# print(encoded_text)
# output = model(**encoded_text)
# print(output)
# converting the raw model output from a PyTorch tensor into a NumPy array :
# scores = output[0][0].detach().numpy()
# print(scores)
# scores = softmax(scores)  # converts a vector of raw scores (logits) into probabilities (0, 1)
# print(scores)
# scores_dict = {'roberta-neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}
# print(scores_dict)
# {'roberta-neg': 0.97635514, 'roberta_neu': 0.020687465, 'roberta_pos': 0.0029573692}


def polarity_scores_roberta(examp):
    encoded_text = tokenizer(examp, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}
    return scores_dict


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):  # calculating and storing the polarity score for each row in a dictionary
    try:
        text = row['Text']
        my_id = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_name = {}
        for key, value in vader_result.items():
            vader_result_name[f'vader_{key}'] = value
        roberta_result = polarity_scores_roberta(text)
        both = vader_result_name | roberta_result  # merge both dictionaries
        res[my_id] = both
    except RuntimeError:
        print(f'broke for id {my_id}')

# Converting the res dictionary to a dataframe. This time res contains VADER and RoBERTa results
results_df = pd.DataFrame(res).T
print(results_df)
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
print(results_df.head())

# 6 - Compare scores between models
# print(results_df.columns)
# visualizing the relationship between sentiment scores predicted by VADER and the RoBERTa model,
# with respect to the review scores :
sns.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg',
                                    'roberta_neu', 'roberta_pos'], hue='Score', palette='tab10')
# plt.show()

# 7 - Review Examples


