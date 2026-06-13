def train_model():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import time
    from sklearn.inspection import permutation_importance
    import numpy as np
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.metrics import recall_score, f1_score
    from sklearn.metrics import confusion_matrix
    import joblib
    from preprocessing import preprocess_data
    import pandas as pd

    df = pd.read_csv("project/data/train.csv")

    X_train_final, X_test_final, y_train, y_test, ct, le, X = preprocess_data(df)

    # SPRINT 2 - Step 2: Train multiple classifiers and evaluate
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import time

    models = {

        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),

    }

    results = []
    trained_models = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train_final, y_train)
        train_time = time.time() - start

        y_tr_pred = model.predict(X_train_final)
        y_te_pred = model.predict(X_test_final)

        train_metrics = {
            'accuracy': accuracy_score(y_train, y_tr_pred),
            'precision': precision_score(y_train, y_tr_pred, zero_division=0),
            'recall': recall_score(y_train, y_tr_pred, zero_division=0),
            'f1': f1_score(y_train, y_tr_pred, zero_division=0)
        }
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_te_pred),
            'precision': precision_score(y_test, y_te_pred, zero_division=0),
            'recall': recall_score(y_test, y_te_pred, zero_division=0),
            'f1': f1_score(y_test, y_te_pred, zero_division=0)
        }

        results.append({
            'model': name,
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'train_time_s': round(train_time, 2)
        })
        trained_models[name] = model


# Use RandomForest model for permutation importance (good for this task)
    best_perm_model = trained_models['RandomForest']

# Calculate permutation importance on test set
    perm_importance = permutation_importance(
        best_perm_model, X_test_final, y_test, 
        n_repeats=10, random_state=42, n_jobs=-1
    )

# Create a DataFrame for easier viewing
    perm_importance_df = pd.DataFrame({
        'feature': X_test_final.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    # SPRINT 3 - Step 1: Feature Engineering (Optimized)

# Define service rating columns
    rating_cols = ['wifi','convenience','booking','gate','food','boarding', 
                'seat','entertainment','onboard_service','leg_room', 
                'baggage','checkin','inflight_service','cleanliness']

# ==================== KEPT FEATURES ====================
# WiFi satisfaction (low satisfaction indicator)
    X['wifi_low'] = (X['wifi'] <= 2).astype(int)

# Service rating variability (std of all service ratings)
    X['rating_std'] = X[rating_cols].std(axis=1)

# ==================== NEW ENGINEERED FEATURES ====================
# 1. Premium Customer: Loyal customers in Business/Business Plus class
    X['premium_customer'] = (
        (X['customer_type'] == 'Loyal Customer') & 
        (X['class'].isin(['Business', 'Business Plus']))
    ).astype(int)

# 2. Comfort Score: Average of comfort-related services
    X['comfort_score'] = X[['seat', 'leg_room', 'cleanliness']].mean(axis=1)

# 3. Digital Experience: Average of tech/entertainment services
    X['digital_experience'] = X[['wifi', 'entertainment']].mean(axis=1)

# ==================== ADDITIONAL SUGGESTED FEATURES ====================
# 4. Service Quality Score: Average of operational services
    service_quality_cols = ['onboard_service', 'inflight_service', 'checkin', 'boarding']
    X['service_quality'] = X[service_quality_cols].mean(axis=1)

# 5. Booking Experience: Average of pre-flight booking & convenience
    X['booking_experience'] = X[['booking', 'convenience']].mean(axis=1)

# 6. Delay Severity Binary: Has significant departure delay (using log transform)
    X['has_significant_delay'] = (X['dep_delay_log'] > np.log1p(10)).astype(int)

# 7. Overall Service Rating Mean (all rating columns)
    X['overall_service_mean'] = X[rating_cols].mean(axis=1)

# Refresh numeric / categorical lists for downstream pipeline
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

# Quick sample of new features
    sample_cols = ['wifi_low', 'rating_std', 'premium_customer', 'comfort_score', 
                'digital_experience', 'service_quality', 'booking_experience', 
                'has_significant_delay', 'overall_service_mean']



    numeric_cols_for_corr = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    corr_matrix = X[numeric_cols_for_corr].corr().abs()

    corr_threshold = 0.85
    high_corr_remove = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
            # Drop second feature by convention
                high_corr_remove.append(col2)

    high_corr_remove = list(set(high_corr_remove))
    X_train_corr = X_train_final.drop(
        columns=[c for c in high_corr_remove if c in X_train_final.columns],
        errors='ignore'
    )

    X_test_corr = X_test_final.drop(
        columns=[c for c in high_corr_remove if c in X_test_final.columns],
        errors='ignore'
    )
    rf_perm = trained_models['RandomForest']

    perm_result = permutation_importance(
        rf_perm,
        X_test_corr,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_importance_df = pd.DataFrame({
        'feature': X_test_corr.columns,
        'importance': perm_result.importances_mean
    }).sort_values('importance', ascending=False)

# Use TOP N instead of mean threshold

    top_n = 10

    selected_by_perm = perm_importance_df.head(top_n)['feature'].tolist()
    
    
    rf_rfe = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    target_features = 12

    rfe = RFE(
        estimator=rf_rfe,
        n_features_to_select=target_features,
        step=2
    )

    rfe.fit(X_train_corr, y_train)

    rfe_selected_features = X_train_corr.columns[
        rfe.support_
    ].tolist()  

    
    feature_scores = {}

    for feat in X_train_corr.columns:

        score = 0

    # Vote 1 → survived correlation filtering
        if feat not in high_corr_remove:
            score += 1

    # Vote 2 → permutation importance
        if feat in selected_by_perm:
            score += 1

    # Vote 3 → RFE
        if feat in rfe_selected_features:
            score += 1

        feature_scores[feat] = score


# Keep features selected by at least 2 methods

    final_selected_features = [
        feat for feat, score in feature_scores.items()
        if score >= 2
    ]

    X_train_selected = X_train_corr[final_selected_features]
    X_test_selected = X_test_corr[final_selected_features]

    reduction = 100 * (
        1 - X_train_selected.shape[1] / X_train_final.shape[1]
    )

    summary_df = pd.DataFrame({
        'feature': list(feature_scores.keys()),
        'votes': list(feature_scores.values())
    }).sort_values(
        'votes',
        ascending=False
    )



    rf_param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=rf_param_dist,
        n_iter=20,
        cv=5,
        scoring='f1',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    rf_random.fit(X_train_selected, y_train)



    rf_grid = {
        'n_estimators': [250, 300, 350],
        'max_depth': [15, 20, 25],
        'min_samples_split': [8, 10, 12],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    grid_search.fit(X_train_selected, y_train)

    best_rf = grid_search.best_estimator_


    pred = best_rf.predict(X_test_selected)

    

    joblib.dump({
        "model": best_rf,
        "transformer": ct,
        "label_encoder": le,
        "selected_features": final_selected_features
    }, "model.joblib")

if __name__ == "__main__":
    train_model()