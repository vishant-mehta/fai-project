import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Read the CSV file
df = pd.read_csv('data/marco.csv')

# Print the shape of the DataFrame
print("DataFrame shape:", df.shape)
print("Column names:", df.columns)
print("Data types:")
print(df.dtypes)

print("Number of duplicate rows:", df.duplicated().sum())

# Get the distribution of query types
print("Query type distribution:")
query_type_counts = df['query_type'].value_counts()
print(query_type_counts)

plt.figure(figsize=(8, 6))
sns.countplot(x='query_type', data=df)
plt.xlabel('Query Type')
plt.ylabel('Count')
plt.title('Query Type Distribution')
plt.xticks(rotation=45)
plt.show()

# Get the distribution of answer lengths
print("Answer length distribution:")
if 'answer' not in df.columns:
    print("Column 'answer' does not exist in the DataFrame.")
df['answer_length'] = df['answers'].apply(eval).apply(lambda x: len(x[0]) if x else 0)
answer_length_counts = df['answer_length'].value_counts()
print(answer_length_counts)

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='answer_length', bins=20)
plt.xlabel('Answer Length')
plt.ylabel('Count')
plt.title('Answer Length Distribution')
plt.show()

print("Average passages per query type:")
passages_per_query_type = df.groupby('query_type')['passages'].apply(lambda x: x.apply(eval).apply(len).mean())
print(passages_per_query_type)

# Get the most common words in the contexts
all_contexts = ' '.join(df['context'])
context_words = all_contexts.split()
context_word_counts = pd.Series(context_words).value_counts()
print("Most common context words:")
print(context_word_counts.head(10))



wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(context_word_counts)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Context Words in the dataset')
plt.show()

df['answer_length'] = df['answers'].apply(eval).apply(lambda x: len(x[0]) if x else 0)


# Check if any passages are missing
missing_passages = df[df['passages'].apply(eval).apply(lambda x: not x)]
print("Number of missing passages:", len(missing_passages))

# Get the distribution of the number of passages per query
print("Distribution of the number of passages per query:")
passages_per_query = df['passages'].apply(eval).apply(len)
print(passages_per_query.describe())

# Visualize the distribution of the number of passages per query
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x=passages_per_query, bins=20)
plt.xlabel('Number of Passages per Query')
plt.ylabel('Count')
plt.title('Distribution of the Number of Passages per Query')
plt.show()

# Get the most common bigrams in the queries
from nltk import ngrams

all_queries = ' '.join(df['query'])
query_bigrams = list(ngrams(all_queries.split(), 2))
bigram_counts = pd.Series(query_bigrams).value_counts()
print("Most common query bigrams:")
print(bigram_counts.head(10))

print("Query length distribution:")
df['query_length'] = df['query'].apply(lambda x: len(x.split()))
print(df['query_length'].describe())

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='query_length', bins=20)
plt.xlabel('Query Length')
plt.ylabel('Count')
plt.title('Query Length Distribution')
plt.show()

all_contexts = ' '.join(df['context'])
context_trigrams = list(ngrams(all_contexts.split(), 3))
trigram_counts = pd.Series(context_trigrams).value_counts()
print("Most common context trigrams:")
print(trigram_counts.head(10))

multiple_answers = df[df['answer'].apply(eval).apply(len) > 1]
print("Number of queries with multiple answers:", len(multiple_answers))

# Get the distribution of the number of selected passages per query
print("Distribution of the number of selected passages per query:")
selected_passages_per_query = df['passages'].apply(eval).apply(lambda x: sum(p['is_selected'] for p in x))
print(selected_passages_per_query.describe())

# Visualize the distribution of the number of selected passages per query
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x=selected_passages_per_query, bins=20)
plt.xlabel('Number of Selected Passages per Query')
plt.ylabel('Count')
plt.title('Distribution of the Number of Selected Passages per Query')
plt.show()