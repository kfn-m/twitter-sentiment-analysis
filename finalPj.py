import pandas as pd
import matplotlib.pyplot as plt  
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Twitter_Data.csv")

df.rename(columns={"clean_text": "text"}, inplace=True)


def clean_text(text):
     text = str(text)
     text = text.lower()
     text = re.sub(r"http\S+", "", text)
     text = re.sub(r"@\w+", "", text)
     text = re.sub(r"[^a-z\s]", "", text)
     return text 

df["clean_text"] = df["text"].apply(clean_text)

df = df.dropna(subset=["text", "category"])

df["text"].isna().sum()

X = df["text"]

y = df["category"]

vectorizer = TfidfVectorizer(max_features=5000 , stop_words='english' ) 

X = vectorizer.fit_transform(df["text"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    
    stratify = y,
    test_size=0.2,
    random_state=42
    )

model = LogisticRegression(max_iter=1000) #max iter مرات التعلم 

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))


examples = [
    "I love this government",
    "This is very bad",
    "This is okay",
    "this comes from cabinet which has scholars like modi smriti and hema time introspect",
    "The new economic policy is a great step forward for the country.",
    "I am so proud of the progress we are making lately!",
    "Absolutely wonderful news, this is what we waited for.",
    "This is a total disaster and a waste of public money.",
    "I am very disappointed with the recent decisions, honestly.",
    "Terrible service and even worse management. Fix this!",
    "The minister scheduled a meeting for next Tuesday at 10 AM.",
    "Today is a cloudy day in the capital city.",
    "The report was published on the official website yesterday."
 ]

examples_vec = vectorizer.transform(examples)

print(model.predict(examples_vec))    