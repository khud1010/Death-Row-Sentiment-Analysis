# %%
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
### Function to process documents
###############################################################################

def clean_doc(doc):
    """
    Tokenizes, removes punctuation, non-alphabetic tokens, short tokens,
    stopwords, and stems the words in the document.
    """
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if len(word) > 3]
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return tokens

###############################################################################
### Functions for Label Encoding
###############################################################################

def One_Hot(variable):
    """
    Encodes categorical variables using one-hot encoding.
    """
    LE = LabelEncoder()
    LE.fit(variable)
    Label1 = LE.transform(variable)
    OHE = OneHotEncoder()
    labels = OHE.fit_transform(Label1.reshape(-1, 1)).toarray()
    return labels, LE, OHE

###############################################################################
### Processing Text into Lists
###############################################################################

#read in class corpus csv into python
data = pd.read_csv('./inmate_last_words.csv')
data["full_name"] = data["First Name"] + " " + data["Last Name"]
#%%
data['Race'] = data['Race'].str.strip()

# Create lists to store text documents titles and body
titles = data['full_name'].tolist()
text_body = data['last_statement'].tolist()

data = data.reset_index()
print("Num after deletion:\n", len(titles))

# Process the text documents
processed_text = [clean_doc(i) for i in text_body]

# Stitch back together individual words to reform body of text
final_processed_text = [' '.join(row) for row in processed_text]

###############################################################################
### Sklearn TFIDF 
###############################################################################

# Call Tfidf Vectorizer
Tfidf = TfidfVectorizer(ngram_range=(1, 1))

# Fit the vectorizer using final processed documents
TFIDF_matrix = Tfidf.fit_transform(final_processed_text)

# Creating dataframe from TFIDF Matrix
matrix = pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names_out(), index=titles)

###############################################################################
### Word Frequency Analysis
###############################################################################

# Flatten the list of processed text to get a single list of all words
all_words = [word for sublist in processed_text for word in sublist]

# Get the frequency of each word
word_freq = Counter(all_words)

# Convert to DataFrame for easy visualization
word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

# Display the top 10 most frequent words
top_words = word_freq_df.head(10)
print("Top 10 Most Frequent Words:\n", top_words)

# Plotting the top 10 most frequent words
plt.figure(figsize=(12, 6))
plt.bar(top_words['Word'], top_words['Frequency'], color='blue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.show()



###############################################################################
### TF-IDF Score Analysis
###############################################################################

# Calculate average TF-IDF score for each word
average_TFIDF = matrix.mean().sort_values(ascending=False)

# Get the highest and lowest scoring words
top_tfidf_words = average_TFIDF.head(10)

print("Top 10 Words by TF-IDF Score:\n", top_tfidf_words)

# Plotting the top 10 highest TF-IDF scoring words
plt.figure(figsize=(12, 6))
plt.bar(top_tfidf_words.index, top_tfidf_words.values, color='green')
plt.xlabel('Words')
plt.ylabel('TF-IDF Score')
plt.title('Top 10 Words by TF-IDF Score')
plt.show()

###############################################################################
### K Means Clustering - TFIDF
###############################################################################

k = 4
km = KMeans(n_clusters=k, random_state=89)
km.fit(TFIDF_matrix)
clusters = km.labels_.tolist()

terms = Tfidf.get_feature_names_out()
Dictionary = {'Doc Name': titles, 'Cluster': clusters, 'Text': final_processed_text}
frame = pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name', 'Text'])

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms_dict = []
cluster_terms = {}
cluster_title = {}

for i in range(k):
    print("Cluster %d:" % i)
    temp_terms = []
    temp_titles = []
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i] = temp_terms
    
    print("Cluster %d titles:" % i, end='')
    temp = frame[frame['Cluster'] == i]
    for title in temp['Doc Name']:
        print(' %s,' % title, end='')
        temp_titles.append(title)
    cluster_title[i] = temp_titles

# Plotting the number of documents per cluster
cluster_counts = frame['Cluster'].value_counts().sort_index()

# Convert the cluster counts to a DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Number of Documents']

# Display the table
print(cluster_counts_df)

plt.figure(figsize=(12, 6))
plt.bar(cluster_counts.index, cluster_counts.values, color='green')
plt.xlabel('Cluster')
plt.ylabel('Number of Documents')
plt.title('Number of Documents per Cluster')
plt.show()

###############################################################################
### Plotting
###############################################################################


mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
dist = 1 - cosine_similarity(TFIDF_matrix)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]

cluster_colors = {0: 'black', 1: 'grey', 2: 'blue', 3: 'firebrick', 4: 'rosybrown', 
                  5: 'red', 6: 'darksalmon', 7: 'sienna', 8: 'orange'}
cluster_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 
                 4: 'Cluster 4', 5: 'Cluster 5', 6: 'Cluster 6', 7: 'Cluster 7', 8: 'Cluster 8'}


df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, color=cluster_colors[name], 
            mec='none', label=cluster_names[name])
    ax.set_aspect('auto')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=False, top=False, labelleft=True)
    
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


print("\n" + "="*50 + "\n")
print('BELOW IS THE CODE FOR RACE AND AGE')
print("\n" + "="*50 + "\n")


# Add Race and Age Information
race_age_data = data[['full_name', 'last_statement', 'Race', 'Age']]

# Define age bins
age_bins = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]

# Sort the Corpus by Race
sorted_by_race = race_age_data.sort_values(by=['Race'])

# Get unique races
races = sorted_by_race['Race'].unique()
#%%
for race in races:
    # Filter data for the current race
    race_data = sorted_by_race[sorted_by_race['Race'] == race]
    
    # Extract titles and text_body for the current race
    race_titles = race_data['full_name'].tolist()
    race_text_body = race_data['last_statement'].tolist()

    # Process the text documents
    processed_text = [clean_doc(i) for i in race_text_body]

    # Stitch back together individual words to reform body of text
    final_processed_text = [' '.join(row) for row in processed_text]

    # Call Tfidf Vectorizer
    Tfidf = TfidfVectorizer(ngram_range=(1, 1))

    # Fit the vectorizer using final processed documents
    TFIDF_matrix = Tfidf.fit_transform(final_processed_text)

    # Perform KMeans clustering
    k = 4  # You can adjust the number of clusters as needed
    km = KMeans(n_clusters=k, random_state=89)
    km.fit(TFIDF_matrix)
    clusters = km.labels_.tolist()

    # Create a dataframe for clustering results
    cluster_results = pd.DataFrame({'Doc Name': race_titles, 'Cluster': clusters})

    # Print top terms per cluster
    print(f"Top terms per cluster for {race}:\n")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = Tfidf.get_feature_names_out()

    for i in range(k):
        print(f"Cluster {i}:")
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(top_terms)

    # Plot the number of documents per cluster
    cluster_counts = cluster_results['Cluster'].value_counts().sort_index()
    # Convert the cluster counts to a DataFrame
    cluster_counts_df = cluster_counts.reset_index()
    cluster_counts_df.columns = ['Cluster', 'Number of Documents']

    # Display the table
    print(cluster_counts_df)
    plt.figure(figsize=(12, 6))
    plt.bar(cluster_counts.index, cluster_counts.values, color='green')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Documents')
    plt.title(f'Number of Documents per Cluster for {race}')
    plt.show()

    # Plotting using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    dist = 1 - cosine_similarity(TFIDF_matrix)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]

    df = pd.DataFrame({'x': xs, 'y': ys, 'label': clusters, 'title': race_titles}) 
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, color=cluster_colors[name], 
                mec='none', label=cluster_names[name])
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=False, top=False, labelleft=True)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'Clusters Visualization for {race}')
    plt.show()

# Sort the Corpus by Age and Bin Ages
race_age_data['Age_Bin'] = pd.cut(race_age_data['Age'], bins=[20, 30, 40, 50, 60, 80], labels=['20-29', '30-39', '40-49', '50-59', '60+'])

# Get unique age bins
age_bins = race_age_data['Age_Bin'].unique()

# Run TF-IDF Procedure for each age bin
for age_bin in age_bins:
    # Filter data for the current age bin
    age_bin_data = race_age_data[race_age_data['Age_Bin'] == age_bin]
    
    # Extract titles and text_body for the current age bin
    age_bin_titles = age_bin_data['full_name'].tolist()
    age_bin_text_body = age_bin_data['last_statement'].tolist()

    # Process the text documents
    processed_text = [clean_doc(i) for i in age_bin_text_body]

    # Stitch back together individual words to reform body of text
    final_processed_text = [' '.join(row) for row in processed_text]

    # Call Tfidf Vectorizer
    Tfidf = TfidfVectorizer(ngram_range=(1, 1))

    # Fit the vectorizer using final processed documents
    TFIDF_matrix = Tfidf.fit_transform(final_processed_text)

    # Perform KMeans clustering
    k = 4  # You can adjust the number of clusters as needed
    try:
        km = KMeans(n_clusters=k, random_state=89)
        km.fit(TFIDF_matrix)
    except:
        print(f'number of samples are too low for {age_bin} year olds')
        continue
    clusters = km.labels_.tolist()

    # Create a dataframe for clustering results
    cluster_results = pd.DataFrame({'Doc Name': age_bin_titles, 'Cluster': clusters})

    # Print top terms per cluster
    print(f"Top terms per cluster for {age_bin}:\n")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = Tfidf.get_feature_names_out()

    for i in range(k):
        print(f"Cluster {i}:")
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(top_terms)

    # Plot the number of documents per cluster
    cluster_counts = cluster_results['Cluster'].value_counts().sort_index()
    # Convert the cluster counts to a DataFrame
    cluster_counts_df = cluster_counts.reset_index()
    cluster_counts_df.columns = ['Cluster', 'Number of Documents']

    # Display the table
    print(cluster_counts_df)
    plt.figure(figsize=(12, 6))
    plt.bar(cluster_counts.index, cluster_counts.values, color='green')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Documents')
    plt.title(f'Number of Documents per Cluster for {age_bin} year olds')
    plt.show()

    # Plotting using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    dist = 1 - cosine_similarity(TFIDF_matrix)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]

    df = pd.DataFrame({'x': xs, 'y': ys, 'label': clusters, 'title': age_bin_titles}) 
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, color=cluster_colors[name], 
                mec='none', label=cluster_names[name])
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=False, top=False, labelleft=True)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'Clusters Visualization for {age_bin} year olds')
    plt.show()

# %%
