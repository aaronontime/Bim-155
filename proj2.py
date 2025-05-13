import numpy as np
import pandas as pd
from icc import icc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# We already get the df for the model fucntion and the predict


def process_data(df):
    df = df.drop(columns=["ID"])
    # turn biological sec and location into numerical values via da
    # labelencoder 
    label_enc_sex = LabelEncoder()
    df['sex'] = label_enc_sex.fit_transform(
        df['sex'])  # turns the female to 0, male to 0
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
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        use_label_encoder=False,
        min_child_weight=20)
    model.fit(train_features_scaled, y_train)
    #print("Model Done fitting ")

    return [model, scaler]

#ask about scaler, will it work like this? Will my resuls be hella off?
#should I do some dimenion reduction? 
def predict(model, df):
    test_data = process_data(df)
    df_pred = pd.read_csv('sample-submission.csv')
    train_features_scaled = model[1].transform(test_data)
    df_pred["Predicted"] = model[0].predict(train_features_scaled)
    # Modify your df_pred DataFrame as needed
    pass
    df_pred.to_csv('predictions.csv', index=False, encoding='utf-8')


# main code

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test_data.csv")
model = train(train_df)
predict(model, test_df)
print("Done")
