#type: ignore


import numpy as np

import joblib


from datasets import train_data

from nepalitokenizer import NepaliTokenizer

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline


np.random.seed(42)


X = train_data['paras']
y = train_data['label']

tokenize = NepaliTokenizer()

tfidf = TfidfVectorizer(tokenizer=tokenize.tokenizer)
pipe_svc = Pipeline([("tfidf", tfidf), ("svc", SVC())])


def tune_nb(tfidf):
    pipe_nb = Pipeline([('tfidf', tfidf), ('NB', MultinomialNB())])
    kfold = StratifiedKFold(n_splits=10).split(
        X, y)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_nb.fit(X[train], y[train])
        score = pipe_nb.score(X[test], y[test])
        scores.append(score)
        print(f'Fold: {k+1: 02d}, Accuracy: {score: .3f}')
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f'CV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')


def tune_svc(pipe_svc):
    param_svc = {
        "svc__C": [1e-2, 1e-3, 1e-1, 1e+2, 1e+3],
        "svc__gamma": [1e+3, 1e+2, 1e-1, 1e-2],
        'svc__kernel': ['rbf', 'linear'],
        'svc__decision_function_shape': ['ovo', 'ovr']
    }

    cv = StratifiedKFold(n_splits=5)

    gs_svc = GridSearchCV(
        pipe_svc,
        param_svc,
        refit=True,
        scoring=make_scorer(f1_score, average='macro'),
        cv=cv,
        n_jobs=-1,
        verbose=5
    )
    svc_model = gs_svc.fit(X, y)
    return svc_model


# tune_nb(tfidf)

# save the svc model and pipeline
svc_model = tune_svc(pipe_svc)

joblib.dump(svc_model, "model_svc.bin")
