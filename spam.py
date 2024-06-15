import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import chardet

def read_data(loc):
    with open(loc, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    
    df = pd.read_csv(
        loc,
        encoding=encoding,
        true_values=['spam'],
        false_values=['ham'],
    )

    # Check if the columns exist before dropping them
    drop_columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    return df.assign(
        text=lambda d: d[d.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    ).rename(
        columns={
            'v1': 'spam'
        }
    )

loc = 'C:\\Users\\user\\Desktop\\Python\\spam\\spam.csv'
df = read_data(loc)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

x_train, x_test, y_train, y_test = train_test_split(df.text, df.spam, stratify=df.spam, test_size=0.4, random_state=40)

clf.fit(x_train, y_train)

while True:
    message = input("Enter a message (or type 'exit' to quit): ")

    if message.lower() == 'exit':
        break

    prediction = clf.predict([message])

    if prediction[0]:
        print("The message is classified as spam.")
    else:
        print("The message is not spam.")
