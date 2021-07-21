import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor



def groupby_plot(df, groupby_cols, agg_col, title, ylab, xlab, agg_type='mean', plot_type='bar', top_n=10):
    '''
    INPUT 
        df - a dataframe holding the requires columns
        groupby_cols - columns to grouby on
        agg_columns - col to aggregate
        agg_type -  Aggreatgation type (mean, sum, count)
        plot_type - type of chart to return (bar, line)
        top_n -  top n values to display
        title - title of the plot
        ylab - ylabel of the plot
        xlab - xlabel of the plot
    OUTPUT
        agg_sr - a Series of aggregated values
        Displays a plot of the aggregation performed
    '''
    
    if plot_type == 'bar':
        if agg_type == 'mean':
            agg_sr = df.groupby(groupby_cols)[agg_col].mean().sort_values(ascending=False)
            
        if agg_type == 'sum':
            agg_sr = df.groupby(groupby_cols)[agg_col].sum().sort_values(ascending=False)
        agg_sr.head(top_n).plot(kind=plot_type, title=title, ylabel=ylab,xlabel=xlab, figsize=(15, 5))
    elif plot_type == 'line':
        if agg_type == 'mean':
            agg_sr = df.groupby(groupby_cols)[agg_col].mean()
            
        if agg_type == 'sum':
            agg_sr = df.groupby(groupby_cols)[agg_col].sum()
        agg_sr.plot(kind=plot_type, title=title, ylabel=ylab,xlabel=xlab, figsize=(15, 5))

    
    return agg_sr


def clean_listing_df(df):
    '''
    INPUT 
        df - Dataframe of Airbnb listings of Seattle 
    OUTPUT
        df - Dataframe with all the cleaning done
    '''

    df = df.replace({"t": 1, "f": 0})
    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%','').astype(float)
    df['host_response_rate'] = df['host_response_rate'].str.replace('%','').astype(float)
    df['price'] = df['price'].str.replace('[^0-9.]+','', regex=True).astype(float)

    return df

def clean_dtype(df):
    '''
    INPUT 
        df - A dataframe
    OUTPUT
        df - Dataframe with 2 unique values changed to int and dropped columns which donot have unique values
    '''

    for col, val in df.nunique(axis=0).iteritems():
            if val == 2:
                df[col] = df[col].astype(int)
            elif val < 2:
                df.drop(columns=[col], inplace=True)

    return df

def fit_linear_model(X_train, X_test, y_train, y_test):
    '''
    INPUT 
        X_train - Matrix of features to train model on
        X_test - Matrix of features to test model on
        y_train - Matrix of target to train model on
        y_test - Matrix of target to test model on
    OUTPUT
        lm_model - Returns trained linear regression model
        test_score - r2 score for test data
        train_score - r2 score for training data

    '''

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit
            
    # Predict and score the model

    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using the model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return lm_model, test_score, train_score

def fit_random_forest(X_train, X_test, y_train, y_test):
    '''
    INPUT 
        X_train - Matrix of features to train model on
        X_test - Matrix of features to test model on
        y_train - Matrix of target to train model on
        y_test - Matrix of target to test model on
    OUTPUT
        rfr_model - Returns trained Random Forest Regression Model
        test_score_r2 - r2 score for test data
        train_score_r2 - r2 score for training data
        test_score_mse - MSE for test data
        train_score_mse - MSE for training data
    '''


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test) 

    #Using a random forest regressor as this gave the best results
    rfr_model = RandomForestRegressor(n_estimators=150,
                                criterion='mse',
                                n_jobs=-1)

    rfr_model.fit(X_train,y_train.squeeze())

    #Getting the predicted values and seeing how well out model scored
    y_train_preds = rfr_model.predict(X_train)
    y_test_preds = rfr_model.predict(X_test)

    train_score_r2  = r2_score(y_train, y_train_preds)
    test_score_r2 = r2_score(y_test, y_test_preds)

    train_score_mse = mean_squared_error(y_train, y_train_preds),
    test_score_mse = mean_squared_error(y_test, y_test_preds)

    print('Random Forest MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_preds),
        mean_squared_error(y_test, y_test_preds)))
    print('Random Forest R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_preds),
        r2_score(y_test, y_test_preds)))

    return rfr_model, train_score_r2, test_score_r2, train_score_mse, test_score_mse