
import json
import csv

# Read JSON data from a file
file_path = "20230914_080417_issue_sharings.json"  
with open(file_path, "r") as file:
    json_data = json.load(file)

# Extract authors and their prompts
data = []

for source in json_data["Sources"]:
    first_conversation = source.get("ChatgptSharing", [{}])[0].get("Conversations", [{}])[0]
    user_prompt = first_conversation.get("Prompt", "Unknown")
    author = source.get("Author", "Unknown")

    # Extract additional relevant information as needed
    # For example, you might want to extract the issue type based on keywords in user_prompt

    data.append({"Author": author, "Prompt": user_prompt})

# CSV file path
csv_file_path = "file.csv"  

# Write data to CSV file
with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Author", "Prompt"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write data rows
    for row in data:
        writer.writerow(row)

import json
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import download
import pandas as pd

# Download NLTK resources if not already downloaded
download("stopwords")
download("punkt")

# Read CSV data
csv_file_path = "file.csv" h
df = pd.read_csv(csv_file_path)

# Text preprocessing functions
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.lower() not in stop_words]


    return " ".join(tokens)

# Apply preprocessing to the "Prompts" column
df["Prompts"] = df["Prompt"].apply(preprocess_text)

# Save the preprocessed DataFrame to a new CSV file
preprocessed_csv_file_path = "dataframe.csv"
df.to_csv(preprocessed_csv_file_path, index=False)

df

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a WordCloud for the entire dataset
all_prompts_text = ' '.join(df['Prompts'])
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_prompts_text)

# Plot the WordCloud
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

import seaborn as sns

# Count the occurrences of each author
author_counts = df['Author'].value_counts()

# Plot a bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=author_counts.index, y=author_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Author')
plt.ylabel('Number of Prompts')
plt.title('Top Authors and Number of Prompts')
plt.show()

# Calculate prompt lengths
df['Prompt_Length'] = df['Prompts'].apply(lambda x: len(x.split()))

# Plot a histogram of prompt lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['Prompt_Length'], bins=30, kde=True)
plt.xlabel('Prompt Length')
plt.ylabel('Frequency')
plt.title('Distribution of Prompt Lengths')
plt.show()

import pandas as pd
import re

# Load dataset
df = pd.read_csv("dataframe.csv") 

# Function to remove special characters from a string
def remove_special_characters(text):
    # Using a regular expression to remove non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

# Apply the function to the 'Prompts' column
df['Prompts'] = df['Prompts'].apply(remove_special_characters)

# Create a new DataFrame with only the 'Prompts' column
cleaned_df = pd.DataFrame(df['Prompts'])

# Save the cleaned prompts to a new CSV file
cleaned_csv_file_path = "onlyprompts.csv"  # Replace with your desired file path
cleaned_df.to_csv(cleaned_csv_file_path, index=False)

import pandas as pd


df = pd.read_csv("dataframe.csv")  # Replace with your file path

# Create a new column for labels
df['Category'] = ''

# Define categories
categories = ["bugs", "theoretical questions", "snippets", "feature requests"]

# Iterate through prompts for labeling
for i, prompt in enumerate(df['Prompt']):
    print(f"\nPrompt {i + 1}/{len(df)}:")
    print(prompt)

    # Prompt user for category input
    print("\nSelect a category:")
    for idx, category in enumerate(categories, start=1):
        print(f"{idx}. {category}")

    while True:
        try:
            category_choice = int(input("Enter the number corresponding to the category: "))
            if 1 <= category_choice <= len(categories):
                break
            else:
                print("Invalid input. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_category = categories[category_choice - 1]
    df.at[i, 'Category'] = selected_category

# Save the labeled dataset
df.to_csv("path/to/your/labeled/dataset.csv", index=False)

import pandas as pd

# Load your dataset
df = pd.read_csv("dataframe.csv")

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform the prompts
X = vectorizer.fit_transform(df['Prompts'])

from sklearn.cluster import KMeans

# Specify the number of clusters (you may need to tune this)
num_clusters = 4

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Display prompts and assigned clusters
print(df[['Prompts', 'Cluster']])

# Save the DataFrame to a new CSV file
output_csv_file_path = "clusterfile.csv"
df.to_csv(output_csv_file_path, index=False)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Assume df is preprocessed DataFrame
X = df['Prompts']
y = df['Label']  # Replace 'Label' with the actual column containing your classes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model: Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
