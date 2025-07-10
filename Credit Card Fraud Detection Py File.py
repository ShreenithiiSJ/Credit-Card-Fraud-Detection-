import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras

def load_data(path='creditcard.csv'):
    df = pd.read_csv(path)
    df = df.dropna()
    return df

def preprocess(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_scaled = StandardScaler().fit_transform(X[['Time','Amount']])
    X[['Time','Amount']] = X_scaled
    pca = PCA(n_components=10, random_state=42)
    P = pca.fit_transform(X)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")
    return P, y

def balance(P, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(P, y)
    return X_res, y_res

def build_autoencoder(input_dim):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(64, activation='relu')(inp)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    z = keras.layers.Dense(8, activation='relu')(x)
    x = keras.layers.Dense(16, activation='relu')(z)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(input_dim, activation=None)(x)
    autoencoder = keras.Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(autoencoder, X):
    history = autoencoder.fit(
        X[y==0], X[y==0],
        epochs=50, batch_size=256,
        validation_split=0.1,
        verbose=2
    )
    return history

def evaluate_autoencoder(autoencoder, X):
    recon = autoencoder.predict(X)
    mse = np.mean(np.power(X - recon, 2), axis=1)
    threshold = np.percentile(mse, 99)
    preds = (mse > threshold).astype(int)
    return preds, threshold

def train_classifiers(X, y):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=skf, scoring='roc_auc')
        print(f"{name} ROC-AUC CV: {scores.mean():.4f} ± {scores.std():.4f}")
        m.fit(X, y)
    return models

def test_eval(models, X_test, y_test):
    for name, m in models.items():
        p = m.predict(X_test)
        print(f"--- {name} ---")
        print(classification_report(y_test, p))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, p))

if __name__ == '__main__':
    df = load_data()
    P, y = preprocess(df)
    X_bal, y_bal = balance(P, y)
    models = train_classifiers(X_bal, y_bal)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )
    test_eval(models, X_test, y_test)

    ae = build_autoencoder(X_train.shape[1])
    train_autoencoder(ae, X_train)
    ae_preds, thresh = evaluate_autoencoder(ae, X_test)
    auc = roc_auc_score(y_test, ae_preds)
    print(f"Autoencoder AUC: {auc:.4f}, threshold={thresh:.4f}")
