import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df=pd.read_csv(r'balanced_df.csv')

X=df['review']
y=df['sentiment']

X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

pipeline=Pipeline([
    ('tfidf',TfidfVectorizer(  ngram_range=(1,3),
        max_features=5000,
        sublinear_tf=True)),
    ('clf',LogisticRegression(class_weight='balanced',
        max_iter=1000))
])

new_negative = [
    'return',
    'retured',
    'refund',
    'broken',
    'not working',
    'poor',
    'bad',
    'terrible',
    'disappointed',
    'I returned this product',
    'Had to return this product',
    'Returned after one day',
    'Asked for a refund',
    'Not worth buying',
    'Product was defective',
    'Returned due to poor quality'
]

new_labels = [0] * len(new_negative)

X_train_e = X_train.tolist() + new_negative
y_train_e = y_train.tolist() + new_labels
 
extra_positve=[
    "not bad",
    "not bad at all",
    "not terrible",
    "not disappointing",
    "not bad for the price",
    "Not bad at all.",
    "Not terrible for the price.",
    "Not disappointed with the purchase.",
    "The receptionist was extremely helpful.",
    "not bad for the price",
    "not bad for it"
]

extra_label=[1]*len(extra_positve)

x_train_ex=X_train_e + extra_positve
y_train_ex=y_train_e + extra_label


pipeline.fit(x_train_ex,y_train_ex)
pred = pipeline.predict(x_test)

path=input(r'Enter file path: ').strip().strip('"').strip("'")

if path.endswith('.csv'):
    reveiw_col=input('Enter reveiw column: ')
    df=pd.read_csv(path).copy()
    df=df.dropna(subset=[reveiw_col])
    df=df.drop_duplicates(subset=[reveiw_col])
    text=df[reveiw_col].astype(str).str.lower().str.strip()

    pred = pipeline.predict(text)

    df['prediction'] = pred
    print(df[[reveiw_col,'prediction']])

elif path.endswith('.txt'):
    with open(path,'r',encoding='utf-8') as f:
        text=[line.strip().lower() for line in f if line.strip()]

    text = list(set(text))

    pred = pipeline.predict(text)

else:
    print('Unsupported file')

