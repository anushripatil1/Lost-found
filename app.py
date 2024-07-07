from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
bootstrap = Bootstrap(app)

# Load emails from CSV
df = pd.read_csv('Mails  - Sheet1 (1).csv')

# Preprocessing function
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing
df['Mails'] = df['Mails'].apply(preprocess)

# Encoding labels
label_dict = {"Placement": 0, "Academic": 1, "Lost and Found": 2}
df['Label'] = df['Label'].map(label_dict)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(df['Mails'])
y_train = df['Label']

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', clf)
])

def classify_email(mail_draft):
    preprocessed_draft = preprocess(mail_draft)
    prediction = pipeline.predict([preprocessed_draft])
    for label, index in label_dict.items():
        if index == prediction[0]:
            return label

@app.route('/', methods=['GET', 'POST'])
def classify_form():
    if request.method == 'POST':
        email_text = request.form['email_text']
        classification = classify_email(email_text)
        return render_template('result.html', classification=classification, email_text=email_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
