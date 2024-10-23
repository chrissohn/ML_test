import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error,roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Load in datasets
train_df = pd.read_csv("C:\\Training_DataSet_modified.csv")
test_df = pd.read_csv("C:\\Test_DataSet_modified.csv")

# Replacing these for the AUC analysis and to prevent overfitting/insufficient learning
replacements = {
    "Limited 4x4": "Limited",
    "Limited 75th Anniversary": "Limited",
    "Limited X": "Limited",
    "75th Anniversary Edition": "75th Anniversary"
}
train_df["Vehicle_Trim"] = train_df["Vehicle_Trim"].replace(replacements)

# Remove any instance where dealer list price or trim are NA in training
train_df = train_df.dropna(subset=["Dealer_Listing_Price", "Vehicle_Trim"])

# Differentiating categorical/numerical columns
train_cat_columns = train_df.select_dtypes(include=["object", "bool"]).columns
train_num_columns = train_df.select_dtypes(exclude=["object", "bool"]).columns

test_cat_columns = test_df.select_dtypes(include=["object", "bool"]).columns
test_num_columns = test_df.select_dtypes(exclude=["object", "bool"]).columns

# Create a record of ListingIDs that will have imputed values
train_imputed_ids = train_df[train_df.isna().any(axis=1)]["ListingID"]
test_imputed_ids = test_df[test_df.isna().any(axis=1)]["ListingID"]

# Convert VehCertified to String as opposed to bool
train_df["VehCertified"] = train_df["VehCertified"].astype(str)
test_df["VehCertified"] = test_df["VehCertified"].astype(str)

# Impute numerical missing values with the median of the values
num_imputer = SimpleImputer(strategy="median")
train_df[train_num_columns] = num_imputer.fit_transform(train_df[train_num_columns])
test_df[test_num_columns] = num_imputer.fit_transform(test_df[test_num_columns])

# Impute categorical missing values with the most frequent value 
cat_imputer = SimpleImputer(strategy="most_frequent")
train_df[train_cat_columns] = cat_imputer.fit_transform(train_df[train_cat_columns])
test_df[test_cat_columns] = cat_imputer.fit_transform(test_df[test_cat_columns])

# Removing NAs as regression cannot deal with them
train_df = train_df.dropna()
test_df = test_df.dropna()

# Removing and storing target columns from training df 
vehicle_trim = train_df["Vehicle_Trim"]
dealer_listing_price = train_df["Dealer_Listing_Price"]
train_df = train_df.drop(columns = ["Vehicle_Trim", "Dealer_Listing_Price"])

# Dropping listingID from the columns to not take this into account in the model
train_num_columns = train_num_columns.drop(["ListingID", "Dealer_Listing_Price"])
train_cat_columns = train_cat_columns.drop("Vehicle_Trim")

# Creating the transformers and preprocessor
enc = OneHotEncoder(sparse_output = False,handle_unknown="ignore")
scaler = StandardScaler()

all_preprocessor = ColumnTransformer(
    [
        ("one_hot_encoder", enc, train_cat_columns),
        ("standard_scaler", scaler, train_num_columns),
    ]
)

random_forest_classifier = make_pipeline(
    all_preprocessor,
    RandomForestClassifier(n_estimators=100, random_state=42)
)

random_forest_regressor = make_pipeline(
    all_preprocessor,
    RandomForestRegressor(n_estimators=100, random_state=42)
)

neural_network_regressor = make_pipeline(
    all_preprocessor,
    MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation="relu", max_iter=1000, random_state=42, solver="adam", learning_rate_init=0.001)
)
neural_network_classifier = make_pipeline(
    all_preprocessor,
    MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation="relu", max_iter=1000, random_state=42, solver="adam", learning_rate_init=0.001)
)

# Function to output df of predicted variables
def prediction(model, train_df, target, original_test):
    # Splitting the training data into training and validation set
    data_train, data_test, target_train, target_test = train_test_split(
    train_df, target, random_state=42
    )

    model.fit(data_train, target_train)
    df = model.predict(original_test)
    return df

# Function to take model, training data, target prediction column and output model accuracy based on target variable
def scores(model, train_df, target):
    # See if it is classification or regression based on dtype
    ## Show R2 in regression
    if target.dtypes ==  "float64":
        data_train, data_test, target_train, target_test = train_test_split(
        train_df, target, random_state=42
        )
        model.fit(data_train, target_train)
        model.predict(data_test)
        mae = mean_absolute_error(target_test, model.predict(data_test))
        mse = mean_squared_error(target_test, model.predict(data_test))
        rmse = mse ** 0.5
        print("Model R² accuracy:", model.score(data_test, target_test))
        print("MAE: ", mae)
        print("RMSE: ", rmse)

    # Show AUC score in classification 
    if target.dtypes ==  "object":
        data_train, data_test, target_train, target_test = train_test_split(
        train_df, vehicle_trim, random_state=42, stratify=vehicle_trim
        )
        # Encode target data with LabelEncoder for AUC score
        le = LabelEncoder()
        le.fit(vehicle_trim)
        target_train_encoded = le.transform(target_train)
        target_test_encoded = le.transform(target_test)

        model.fit(data_train, target_train_encoded)
        auc_y = model.predict_proba(data_test)
        print("AUC score:", roc_auc_score(target_test_encoded, auc_y, multi_class="ovr"))
    return 

def outputs(model_regressor, model_classifier):
    # Creating target variable predictions
    trim_prediction = prediction(model_classifier, train_df, vehicle_trim, test_df)
    price_prediction = prediction(model_regressor, train_df, dealer_listing_price, test_df)
    predicted_test_df = test_df
    predicted_test_df["Vehicle_Trim_Predicted"] = trim_prediction
    predicted_test_df["Dealer_Listing_Price_Predicted"] = price_prediction
    test_output = predicted_test_df[["ListingID", "Vehicle_Trim_Predicted", "Dealer_Listing_Price_Predicted"]]
    return test_output

random_forest_output = outputs(random_forest_regressor, random_forest_classifier)
random_forest_output.to_csv("C:\\random_forest_output.csv", index=False, header=False)

neural_network_output = outputs(neural_network_regressor, neural_network_classifier)
neural_network_output.to_csv("C:\\neural_network_output.csv", index=False, header=False)

# Calculating accuracy scores 
scores(random_forest_regressor, train_df, dealer_listing_price)
scores(random_forest_classifier, train_df, vehicle_trim)
scores(neural_network_regressor, train_df, dealer_listing_price)
scores(neural_network_classifier, train_df, vehicle_trim)