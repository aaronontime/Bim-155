import numpy as np
import pandas as pd
from icc import icc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# We already get the df for the model fucntion and the predict


def process_data(df):
    df = df.drop(columns=["ID"])
    # turn biological sec and location into numerical values via da
    # labelencoder \
    label_enc_sex = LabelEncoder()
    df['sex'] = label_enc_sex.fit_transform(
        df['sex'])  # turns the female to 0, male to 1
    label_enc_site = LabelEncoder()
    # turns the site to 0 for left, 1 for right.
    df['site'] = label_enc_site.fit_transform(df['site'])
    return df


def get_features_and_labels(train_data):
    train_feature = train_data.drop(columns=["Expected"])
    train_label = train_data["Expected"]
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_feature)
    return train_features_scaled, train_label, scaler


def train(df):
    train_df = process_data(df)
    train_features_scaled, y_train, scaler = get_features_and_labels(train_df)
    model = XGBClassifier(objective="multi:softprob", num_class=3,
                          n_estimators=50, max_depth=6, learning_rate=0.1,
                          eval_metric="mlogloss", use_label_encoder=False)
    model.fit(train_features_scaled,y_train)
    print("Model Done fitting ")
    return [model, scaler]


def predict(model, df):
    test_data = process_data(df)
    df_pred = pd.read_csv('sample-submission.csv')
    train_features_scaled = model[1].transform(test_data)
    df_pred["Predicted"] = model[0].predict(train_features_scaled)
    # Modify your df_pred DataFrame as needed
    pass
    df_pred.to_csv('predictions.csv', index=False, encoding='utf-8')

from sklearn.metrics import cohen_kappa_score

def cross_val(df, n_splits):  # use train csv
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=20)
    cross_val_df = process_data(df)
    train_features_scaled, train_label, scaler = get_features_and_labels(cross_val_df)
    accuracies = []
    kappa_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_features_scaled)):
        fold_train_features, validate_features = (train_features_scaled[train_idx], train_features_scaled[test_idx])
        fold_train_labels, validate_labels = (train_label[train_idx], train_label[test_idx])
        # 15,6,0.6
        model = XGBClassifier(objective="multi:softprob", num_class=3,
                              n_estimators=100, max_depth=6, learning_rate=0.1,
                              eval_metric="mlogloss", use_label_encoder=False, min_child_weight = 20)
        model.fit(fold_train_features, fold_train_labels)
        
        y_pred_val = model.predict(validate_features)
        
        # Accuracy score for validation
        accuracy = accuracy_score(validate_labels, y_pred_val)
        accuracies.append(accuracy)
        
        # Cohen's Kappa Score
        kappa = cohen_kappa_score(validate_labels, y_pred_val)
        kappa_scores.append(kappa)
        
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Cohen's Kappa: {kappa:.4f}")
    
    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Cohen's Kappa: {np.mean(kappa_scores):.4f}")
    return kappa_scores

from sklearn.decomposition import PCA

def reduce_dimensionality(features):
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca

# main code

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test_data.csv")
#model = train(train_df)
#predict(model, test_df)
train_df = reduce_dimensionality(train_df)
cross_val(train_df,5)
print("Done")
