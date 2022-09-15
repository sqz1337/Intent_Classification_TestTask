import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import warnings
import numpy as np
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv(r'C:\Users\Amaterasu\Downloads\dataset_intent_detection.csv', names=['classification','intent'])
df['intent'] = df['intent'].str.replace('-','')

print(nltk.corpus.stopwords.words('russian'))
lst_stopwords = nltk.corpus.stopwords.words('russian')
new_stopwords = ['что','это','в','и','очень','просто','как','вас','почему']
lst_stopwords.extend(new_stopwords)
remove_words = ['больше','не','где', 'когда','все','нет','всегда','иногда','что','можно','как','куда']
for w in remove_words:
    lst_stopwords.remove(w)


def utils_preprocess_text(text, flg_stemm=True, flg_lemm=True, lst_stopwords=None):
    ## Обработаем текст (преобразуем в нижний регистр и удалим знаки препинания и лишние символы)
    text = re.sub(r'[^\*\w\s]', '', str(text).lower().strip())

    ## Токенизируем (Преобразуем из строки в список)
    lst_text = text.split()
    ## Уберем стоп-слова
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    ## Стемминг (уберем окончания)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Лемматизация (преобразуем слова в изначальную форму)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## вернемся к строке из списка
    text = ' '.join(lst_text)
    return text

df['text_clean'] = df['intent'].apply(lambda x:
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,
          lst_stopwords=lst_stopwords))

X = df.text_clean
y = df.classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

pipe_sgd = Pipeline([
           ('vect', TfidfVectorizer(max_features=5000,ngram_range=(1,2))),
           ('clf', SGDClassifier(loss='modified_huber')),
])
pipe_sgd.fit(X_train, y_train)
y_pred = pipe_sgd.predict(X_test)
predicted_prob = pipe_sgd.predict_proba(X_test)

print('accuracy %s',  accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
predicted_prob = pipe_sgd.predict_proba(X_test)


pickle_file_name = "pipe_sgd.pkl"

with open(pickle_file_name, 'wb') as file:
    pickle.dump(pipe_sgd, file)
input_data = ['Прекрасно']
with open(pickle_file_name, 'rb') as file:
    pk_model = pickle.load(file)

y_predict = pk_model.predict_proba(input_data)

n = 3

probas = pk_model.predict_proba(input_data)
top_n_lables_idx = np.argsort(-probas, axis=1)[:, :n]
top_n_probs = np.round(-np.sort(-probas), 3)[:, :n]
top_n_labels = [pk_model.classes_[i] for i in top_n_lables_idx]

results = list(zip(top_n_labels, top_n_probs))
print(results)
pd.DataFrame(results)
