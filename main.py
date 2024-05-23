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
print(vaders.head())

# Plot VADER results :
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()
