import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk
nltk.download('punkt')


# 1 - Loading the data
df = pd.read_csv('amazon-fine-food-reviews.csv')
# print(df.head())

text_first_row = df['Text'].values[0]
# print(text_first_row)

# print(df.shape)

df = df.head(500)  # reducing dataset size
# print(df.shape)

# 2 - Quick EDA
ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars',
                                                  figsize=(10, 5))
ax.set_xlabel('Review Stars')
# plt.show()

# 3 - Basic NLTK
example = df['Text'][50]
print(example)

tokens = nltk.word_tokenize(example)  # Tokenizers divide strings into lists of substrings
# print(tokens[:10])  # ['This', 'oatmeal', 'is', 'not', 'good', '.', 'Its', 'mushy', ',', 'soft']

tagged = nltk.pos_tag(tokens)
# assigns a part-of-speech tag to each word indicating
# its grammatical category and function in the sentence



