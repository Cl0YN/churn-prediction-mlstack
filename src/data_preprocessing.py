import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(file_path):
    data = pd.read_excel(file_path)
    data = data.drop(['CustomerID', 'Churn Reason', 'Country', 'State', 'Count', 'City'], axis=1)
    return data

def preprocess_data(data):
    data = data.drop(['Latitude', 'Longitude', 'Lat Long', 'Zip Code'], axis=1)
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category')
    data = data.drop(['Churn Label', 'Churn Score'], axis=1)
    cols_1 = ['Online Security', 'Device Protection', 'Tech Support', 'Streaming Movies', 'Streaming TV', 'Online Backup']
    for col in cols_1:
        data[col] = data[col].apply(lambda x: str(x).replace('No internet service', 'No')).astype('category')
    data['Multiple Lines'] = data['Multiple Lines'].apply(lambda x: str(x).replace('No phone service', 'No')).astype('category')
    data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
    data = data.dropna()
    return data

def split_data(data, target_column='Churn Value', test_size=0.2, random_state=42):

    y = data[target_column]
    X = data.drop([target_column], axis=1)
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)

def upsample_minority_class(X_train, y_train):

    churned = X_train[y_train == 1]
    not_churned = X_train[y_train == 0]
    y_churned = y_train[y_train == 1]
    y_not_churned = y_train[y_train == 0]
    if len(churned) < len(not_churned):
        churned_upsampled, y_churned_upsampled = resample(
            churned, y_churned, replace=True, n_samples=len(not_churned), random_state=42
        )
        X_train = pd.concat([not_churned, churned_upsampled])
        y_train = pd.concat([y_not_churned, y_churned_upsampled])
    return X_train, y_train

def get_preprocessor(X_train):
    cat_cols = X_train.select_dtypes(include='category').columns.tolist()
    num_cols = X_train.select_dtypes(exclude='category').columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
    ])
    return preprocessor