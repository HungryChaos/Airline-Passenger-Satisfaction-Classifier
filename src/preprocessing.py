def preprocess_data(df):
    import pandas as pd
# Drop redundant columns
    df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True, errors='ignore')

# Rename columns to simple one-word terms
    df.columns = [
        'gender', 'customer_type', 'age', 'travel_type', 'class', 'distance',
        'wifi', 'convenience', 'booking', 'gate', 'food', 'boarding',
        'seat', 'entertainment', 'onboard_service', 'leg_room', 'baggage',
        'checkin', 'inflight_service', 'cleanliness', 'dep_delay', 'arr_delay',
        'satisfaction'
    ]

# Select raw features into X and the target into y
    features = [
        'age', 'distance', 'dep_delay', 'arr_delay',
        'gender', 'customer_type', 'travel_type', 'class',
        'wifi', 'convenience', 'booking', 'gate', 'food', 'boarding', 
        'seat', 'entertainment', 'onboard_service', 'leg_room', 
        'baggage', 'checkin', 'inflight_service', 'cleanliness'
    ]

    X = df[features].copy()
    y = df['satisfaction'].copy() # Keep as text for now, we'll encode it separately or as part of y selection
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    # choosing median to fill na values.
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    import numpy as np

# Apply Log Transformation to handles skewed delay data right on X
    X['dep_delay_log'] = np.log1p(X['dep_delay'])
    X['arr_delay_log'] = np.log1p(X['arr_delay'])

# Drop the old delay columns since we have the log versions
    X.drop(['dep_delay', 'arr_delay'], axis=1, inplace=True)

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

# Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

# Refresh the column lists from the fresh feature frame
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

# Split before fitting the transformer
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a single ColumnTransformer for numeric and categorical columns
    ct = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), num_cols),
            ('cat_encoder', OrdinalEncoder(), cat_cols)
        ]
    )

# Fit on train and transform both sets
    X_train_transformed = ct.fit_transform(X_train)
    X_test_transformed = ct.transform(X_test)

    # Convert back to DataFrames for easier inspection
    X_train_final = pd.DataFrame(X_train_transformed, columns=ct.get_feature_names_out())
    X_test_final = pd.DataFrame(X_test_transformed, columns=ct.get_feature_names_out())
    
    return (
        X_train_final,
        X_test_final,
        y_train,
        y_test,
        ct,      # save transformer
        le,      # save label encoder
        X        # return original features for feature engineering
    )