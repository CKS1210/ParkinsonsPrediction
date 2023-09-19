import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_clean_data():
    data = pd.read_csv("data/parkinsons_data.csv")
    data = data.drop(['name'], axis=1)
    return data

def create_model(data):
    X = data.drop(['status'], axis=1)
    y = data['status']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y, test_size=0.2, random_state=42)

     # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print("Accuracy of the model", accuracy_score(y_test, y_pred))
    print("Classification Report of the model \n", classification_report(y_test, y_pred))

    return model, scaler


def main():
    data = get_clean_data()

    model,scaler = create_model(data)

    with open("model/model.pkl","wb") as f:
        pickle.dump(model,f)
    with open("model/scaler.pkl","wb") as f:
        pickle.dump(scaler,f)

if __name__ == '__main__':
    main()