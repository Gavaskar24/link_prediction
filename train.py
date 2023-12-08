from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_model(df, scaler):

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df.drop('Label', axis=1))
    X_val_scaled = scaler.transform(val_df.drop('Label', axis=1))

    log_reg = LogisticRegression(random_state=42, penalty='l2', C=1.0)
    log_reg.fit(X_train_scaled, train_df['Label'])

    val_probs = log_reg.predict_proba(X_val_scaled)[:, 1]
    auc_roc = roc_auc_score(val_df['Label'], val_probs)
    print(f"AUC-ROC on Validation Set: {auc_roc}")

    return log_reg

def train_grid_model(df, scaler):
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train_scaled = scaler.fit_transform(train_df.drop('Label', axis=1))
    y_train = train_df['Label']

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(random_state=42, penalty='l2', max_iter=1000), param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train_scaled, y_train)
    best_clf = grid_search.best_estimator_

    return best_clf

import torch
import torch.nn as nn
import torch.optim as optim

class BinaryNN(nn.Module):
    def __init__(self, input_dim):
        super(BinaryNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader, TensorDataset

class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=10, lr=0.01):
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        input_dim = X.shape[1]
        self.model_ = BinaryNN(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y[:, None], dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                predictions = self.model_(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        return self

    # def predict_proba(self, X):
    #     check_is_fitted(self)
    #     X = check_array(X)
        
    #     with torch.no_grad():
    #         X_tensor = torch.tensor(X, dtype=torch.float32)
    #         probas = self.model_(X_tensor).numpy()
        
    #     return probas
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            pos_probs = self.model_(X_tensor).numpy()
            
        neg_probs = 1 - pos_probs
        
        # return np.vstack([neg_probs, pos_probs]).T
        return np.hstack([neg_probs, pos_probs])

    
import xgboost as xgb

# model_xgb = xgb.XGBClassifier(
#     objective='binary:logistic', 
#     eval_metric='auc', 
#     learning_rate=0.1, 
#     n_estimators=100, 
#     max_depth=5, 
#     random_state=42
# )

from sklearn.svm import SVC
# model_svm = SVC(probability=True, kernel='linear', C=1.0, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
# model_knn = KNeighborsClassifier(n_neighbors=5)

from sklearn.naive_bayes import GaussianNB
# model_nb = GaussianNB()


# def train_ensemble_model(df, scaler, best_clf):
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
#     X_train_scaled = scaler.fit_transform(train_df.drop('Label', axis=1))
#     X_val_scaled = scaler.transform(val_df.drop('Label', axis=1))
#     y_train = train_df['Label']
#     y_val = val_df['Label']
    
#     model_lr = LogisticRegression(random_state=42, penalty='l2', C=best_clf.C)
#     # model_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
#     model_nn = TorchNNClassifier(epochs=10, lr=0.01)
#     model_svm = SVC(probability=True, random_state=42)
#     model_knn = KNeighborsClassifier(n_neighbors=5)
#     model_nb = GaussianNB()
#     model_xgb = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
    
#     ensemble_model = VotingClassifier(estimators=[
#         ('lr', model_lr), 
#         # ('rf', model_rf), 
#         ('nn', model_nn), 
#         ('svm', model_svm), 
#         ('knn', model_knn), 
#         ('nb', model_nb), 
#         ('xgb', model_xgb)
#     ], voting='soft')
    
#     ensemble_model.fit(X_train_scaled, y_train)
    
#     # models = [model_lr, model_rf, model_nn, model_svm, model_knn, model_nb, model_xgb, ensemble_model]
#     models = [model_lr, model_nn, model_svm, model_knn, model_nb, model_xgb, ensemble_model]
#     model_names = ['Logistic Regression', 'Random Forest', 'Neural Network', 'SVM', 'KNN', 'Naive Bayes', 'XGBoost', 'Ensemble']
    
#     for model, name in zip(models, model_names):
#         val_probs = model.predict_proba(X_val_scaled)[:, 1]
#         auc_roc = roc_auc_score(y_val, val_probs)
#         print(f"AUC-ROC of {name} on Validation Set: {auc_roc}")
    
#     return ensemble_model

def train_ensemble_model(df, scaler, best_clf):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(train_df.drop('Label', axis=1))
    X_val_scaled = scaler.transform(val_df.drop('Label', axis=1))
    y_train = train_df['Label']
    y_val = val_df['Label']
    
    model_lr = LogisticRegression(random_state=42, penalty='l2', max_iter=1000, C=best_clf.C)
    model_nn = TorchNNClassifier(epochs=10, lr=0.01)
    model_svm = SVC(probability=True, random_state=42)
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_nb = GaussianNB()
    model_xgb = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
    
    models = [model_lr, model_nn, model_svm, model_knn, model_nb, model_xgb]
    for model in models:
        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"Error fitting model {type(model).__name__}: {e}")
    
    ensemble_model = VotingClassifier(estimators=[
        ('lr', model_lr), 
        ('nn', model_nn), 
        ('svm', model_svm), 
        ('knn', model_knn), 
        ('nb', model_nb), 
        ('xgb', model_xgb)
    ], voting='soft')
    
    try:
        ensemble_model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Error fitting ensemble model: {e}")
    
    models.append(ensemble_model)
    model_names = ['Logistic Regression', 'Neural Network', 'SVM', 'KNN', 'Naive Bayes', 'XGBoost', 'Ensemble']
    
    for model, name in zip(models, model_names):
        try:
            val_probs = model.predict_proba(X_val_scaled)[:, 1]
            auc_roc = roc_auc_score(y_val, val_probs)
            print(f"AUC-ROC of {name} on Validation Set: {auc_roc}")
        except Exception as e:
            print(f"Error evaluating model {name}: {e}")
    
    return ensemble_model

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def train_xgboost_model(df, scaler):
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])

    X_train_scaled = scaler.fit_transform(train_df.drop('Label', axis=1))
    X_val_scaled = scaler.transform(val_df.drop('Label', axis=1))

    dtrain = xgb.DMatrix(X_train_scaled, label=train_df['Label'])
    dval = xgb.DMatrix(X_val_scaled, label=val_df['Label'])

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42,
        'scale_pos_weight': 1,
        'learning_rate': 0.1,
        'max_depth': 6,
    }

    num_rounds = 100
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10, verbose_eval=10)

    print("Distribution of Labels in Validation Set:")
    print(val_df['Label'].value_counts())

    val_probs = bst.predict(dval)
    auc_roc = roc_auc_score(val_df['Label'], val_probs)
    print(f"AUC-ROC on Validation Set: {auc_roc}")

    return bst