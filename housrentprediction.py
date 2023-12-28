import pandas as pd 
import numpy as np
import json
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import StackingRegressor
from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor




# drop missing values
def DropMissing(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

# Missing Ratio
def missingratio(all_data):
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    print(missing_data.head(20))

# Removing Duplicates
def removeDup(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# Remove Outliers
def RemoveOutliers(df):

    # Removing outliers rent above 40000
    df=df[df["rent"] <= 40000].reset_index(drop=True)

    # Removing outliers rent above 24000 and lease_type is BACHELOR
    index_names = df[(df['rent']>24000) & (df['lease_type']=='BACHELOR')].index
    df.drop(index_names, inplace = True)

    # Removing outliers rent above 37000 and GYM is 0
    index_names = df[(df['rent']>37000) & (df['gym']==0)].index
    df.drop(index_names, inplace = True)

    # Removing outliers rent above 27000 and furnishing is NOT_FURNISHED
    index_names = df[(df['rent']>27000) & (df['furnishing'] == 'NOT_FURNISHED')].index
    df.drop(index_names, inplace = True)

    # Removing outliers rent above 27000 and parking is TWO_WHEELER
    index_names = df[(df['rent']>27000) & (df['parking'] == 'TWO_WHEELER')].index
    df.drop(index_names, inplace = True)
    # Removing outliers rent above 27000 and parking is NONE
    index_names = df[(df['rent']>30000) & (df['parking'] == 'NONE')].index
    df.drop(index_names, inplace = True)

    # Removing outliers property_size above 4200
    df=df[df["property_size"] <= 4200].reset_index(drop=True)

    # Removing outliers property_age above 150
    df=df[df["property_age"] <= 150].reset_index(drop=True)

    # Removing cup_board values above 11.0
    index_names = df[df['cup_board'] >11.0].index
    df.drop(index_names, inplace = True)

    # Removing floor values above 16.0
    index_names = df[df['floor'] >16.0].index
    df.drop(index_names, inplace = True)

    # Removing outliers from total_floor based on rent 
    index_names = df[(df['rent']<2000) & (df['total_floor'] == 18)].index
    df.drop(index_names, inplace = True)
    index_names = df[(df['rent']<20000) & (df['total_floor'] == 21)].index
    df.drop(index_names, inplace = True)
    index_names = df[(df['rent']<30000) & (df['total_floor'] == 24)].index
    df.drop(index_names, inplace = True)
    index_names = df[(df['rent']<20000) & (df['total_floor'] == 17)].index
    df.drop(index_names, inplace = True)
    index_names = df[(df['rent']<13000) & (df['total_floor'] == 10)].index
    df.drop(index_names, inplace = True)
    index_names = df[(df['rent']<13000) & (df['total_floor'] == 9)].index
    df.drop(index_names, inplace = True)

    # Removing outliers from building_tyoe based on rent
    index_names = df[(df['rent']>30000) & (df['building_type'] == 'IF')].index
    df.drop(index_names, inplace = True)
    index_names = df[(df['rent']>32000) & (df['building_type'] == 'IH')].index
    df.drop(index_names, inplace = True)
    index_names = df[(df['rent']>32000) & (df['building_type'] == 'GC')].index
    df.drop(index_names, inplace = True)

    return df

def DataEncoding(df):
    # Encoding the values type, lease_type, furnishing, parking, water_supply, facing, building_type
    df['type']= df['type'].map({"RK1":0,"BHK1":1,"1BHK1":1,"BHK2":2,"bhk2":2,"BHK3":3,"bhk3":3,"BHK4":4,"BHK4PLUS":5})
    df["lease_type"]=df["lease_type"].map({"BACHELOR":0,"ANYONE":1,"FAMILY":2,"COMPANY":3})
    df["furnishing"]=df["furnishing"].map({"NOT_FURNISHED":0,"SEMI_FURNISHED":1,"FULLY_FURNISHED":2})
    df["parking"]=df["parking"].map({"NONE":0,"TWO_WHEELER":1,"FOUR_WHEELER":2,"BOTH":3})
    df["water_supply"]=df["water_supply"].map({"BOREWELL":2,"CORP_BORE":1,"CORPORATION":0})
    df["facing"]=df["facing"].map({"E":2,"W":3,"S":1,"N":0,"NE":4,"SE":5,"NW":6,"SW":7})
    df["building_type"]=df["building_type"].map({"IF":1,"AP":3,"IH":2,"GC":0})
    return df

#Feature Selection
def FeatureSelectio(df):  
    rentaldf=df.drop(['id','activation_date','locality','amenities'],axis=1)
    return rentaldf

#splitting data
def split_data_train(rentaldf):
    X=rentaldf.drop('rent',axis=1)
    y=rentaldf['rent']
    return X,y


# Load the dataset
def ProcessData(df):
    df = DropMissing(df)
    df = removeDup(df)
    df = DataEncoding(df)
    df = FeatureSelectio(df)

    return df


# Getting and processing training data
df_train = ProcessData(pd.read_excel(r"C:\Users\kiran\Downloads\House_Rent_Train.xlsx"))
X, y=split_data_train(df_train)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# fit a Linear Regression Model
def LinearRegressionModel(X_train, X_test, y_train, y_test):
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train,)
    score=regressor.score(X_train,y_train)
    print(1-score)
    # Getting testing data
    #X_test = ProcessData(pd.read_excel(r"C:\Users\kiran\Downloads\House_Rent_Test.xlsx"))

    # make predictions
    y_pred = regressor.predict(X_test)

    # calculate R-squared
    print("Liner Regression R-squared: {}".format(regressor.score(X_test, y_test)))

    # calculate root mean squared error (RMSE)
    lin_mse = mean_squared_error(y_pred, y_test)
    lin_rmse = np.sqrt(lin_mse)
    print("Liner Regression RMSE: {}".format(lin_rmse))

    # calculate mean absolute error (MAE)
    lin_mae = mean_absolute_error(y_pred, y_test)
    print("Liner Regression MAE: {}".format(lin_mae))


#Random Forest Model
def RandomForestModel(X_train, X_test, y_train, y_test):    
    
    rf = RandomForestRegressor(random_state=0, max_depth= 22, n_estimators= 38)

    # fit the model
    rf.fit(X_train, y_train)

    
    # calculate R-squared
    print("Random Forest R-squared: {}".format(rf.score(X_test, y_test)))

    # make predictions 
    y_pred = rf.predict(X_test)
    print(y_pred)
    #y_test = np.expm1(y_test)
    print(y_test)
    
    # calculate root mean squared error (RMSE)
    forest_mse = mean_squared_error(y_test, y_pred)
    forest_rmse = np.sqrt(forest_mse)
    print("Random Forest Regression RMSE: {}".format(forest_rmse))


    # calculate mean absolute error (MAE)
    forest_mae = mean_absolute_error(y_test, y_pred)
    print("Random Forest Regression MAE: {}".format(forest_mae))

    return rf


def GridSearchCVModel(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor(random_state=0)
    params = {'max_depth': list(range(20, 30, 2)), 'n_estimators': list(range(30, 40, 2))}
    rf = RandomForestRegressor(random_state=0)
    forest_reg = GridSearchCV(rf, params, cv=5)

    forest_reg.fit(X_train, y_train)
    # best estimator learned through GridSearch
    forest_reg.best_estimator_
    # best parameter values
    print("Best Params:", forest_reg.best_params_)
    # best CV score
    print("Best CV Score:", forest_reg.best_score_)
    # make predictions
    y_pred = forest_reg.predict(X_test)
    # calculate R-squared
    print("Random Forest R-squared: {}".format(forest_reg.score(X_test, y_test)))
    
    # calculate root mean squared error (RMSE)
    forest_mse = mean_squared_error(y_pred, y_test)
    forest_rmse = np.sqrt(forest_mse)
    print("Random Forest Regression RMSE: {}".format(forest_rmse))
   
    # calculate mean absolute error (MAE)
    forest_mae = mean_absolute_error(y_pred, y_test)
    print("Random Forest Regression MAE: {}".format(forest_mae))
    importance = forest_reg.best_estimator_.feature_importances_

    # get feature importances
    feature_indexes_by_importance = importance.argsort()
    for index in feature_indexes_by_importance:
        print('{} : {}'.format(X_train.columns[index], (importance[index] )))


    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    feat_importances = pd.Series(forest_reg.best_estimator_.feature_importances_, index=X_train.columns)
    feat_importances.sort_values().plot(kind="barh", color="Green")
    # feat_importances.nlargest(20).plot(kind='barh') # top 20 features only
    plt.show()


def GradiantBoostModel(X_train, X_test, y_train, y_test):
    # import GradientBoostingRegressor
    gbreg = GradientBoostingRegressor(random_state=0)
    gbreg.fit(X_train, y_train)
    y_pred = gbreg.predict(X_test)
    # calculate R-squared
    print("Gradient Boosting Regressor R-squared: {}".format(gbreg.score(X_test, y_test)))

    # calculate root mean squared error (RMSE)
    gbr_mse = mean_squared_error(y_pred, y_test)
    gbr_rmse = np.sqrt(gbr_mse)
    print("Gradient Boosting Regressor Regression RMSE: {}".format(gbr_rmse))

    # calculate mean absolute error (MAE)
    gbr_mae = mean_absolute_error(y_pred, y_test)
    print("Gradient Boosting Regressor Regression MAE: {}".format(gbr_mae))


def DesicionTreeModel(X_train, X_test, y_train, y_test):
    # create a regressor object 
    regressor = DecisionTreeRegressor(random_state = 0)  
    
    # fit the regressor with X and Y data 
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    # calculate R-squared
    print("desion tree regression R-squared: {}".format(regressor.score(X_test, y_test)))

    # calculate root mean squared error (RMSE)
    dtr_mse = mean_squared_error(y_pred, y_test)
    dtr_rmse = np.sqrt(dtr_mse)
    print("desion tree regression Regression RMSE: {}".format(dtr_rmse))

    # calculate mean absolute error (MAE)
    dtr_mae = mean_absolute_error(y_pred, y_test)
    print("desion tree regression Regression MAE: {}".format(dtr_mae))

def NeuralNetworkModel(X_train, X_test, y_train, y_test):
    regressor = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    # calculate R-squared
    print("MLPregression R-squared: {}".format(regressor.score(X_test, y_test)))

    # calculate root mean squared error (RMSE)
    dtr_mse = mean_squared_error(y_pred, y_test)
    dtr_rmse = np.sqrt(dtr_mse)
    print("MLPregression Regression RMSE: {}".format(dtr_rmse))

    # calculate mean absolute error (MAE)
    dtr_mae = mean_absolute_error(y_pred, y_test)
    print("MLPregression Regression MAE: {}".format(dtr_mae))



rf = RandomForestModel(X_train, X_test, y_train, y_test)

# giving ouput for user input
df_test = (pd.read_excel(r"C:\Users\kiran\Downloads\House_Rent_Test.xlsx"))
df_test = DataEncoding(df_test)
df_test = FeatureSelectio(df_test)


def GetOutput(df_test, rf):    
    
    y_pred = rf.predict(df_test)
    Output = pd.DataFrame(y_pred)
    Output.rename(columns = {0:'rent'}, inplace = True)
    Output.to_excel("Rent_pred.xlsx", sheet_name="Prediction", index=False)
    print(Output)
    

GetOutput(df_test,rf)   
