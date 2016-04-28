#! /usr/bin/env python

import os
import sys
import math
import pandas as pd

# Import a convenience function to split the sets.
from sklearn.cross_validation import train_test_split

# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor


firstYearInTraining = 1990
firstYearInValidation = 2012

# Read in data to dataframe from csv file
df = pd.read_csv('NationalNames.csv')

# Focus only on recent times
df = df[ df['Year']>=firstYearInTraining ]

# Determine rank for given year and gender
df.loc[:,'Rank'] = df.groupby(['Year', 'Gender'])['Count'].rank(ascending=False)

# Calculate the previous years' rank and count
# These will be used as features to predict future behavior
# TODO: add in google trends data
df.loc[:,'m1Rank'] = df.groupby(['Name','Gender'])['Rank'].shift(1)
df.loc[:,'m1Count'] = df.groupby(['Name','Gender'])['Count'].shift(1)

df.loc[:,'m2Rank'] = df.groupby(['Name','Gender'])['Rank'].shift(2)
df.loc[:,'m2Count'] = df.groupby(['Name','Gender'])['Count'].shift(2)

df.loc[:,'m3Rank'] = df.groupby(['Name','Gender'])['Rank'].shift(3)
df.loc[:,'m3Count'] = df.groupby(['Name','Gender'])['Count'].shift(3)

df.loc[:,'m4Rank'] = df.groupby(['Name','Gender'])['Rank'].shift(4)
df.loc[:,'m4Count'] = df.groupby(['Name','Gender'])['Count'].shift(4)


# Gives hotness value based on year and previous year value
def hotness( count, previousCount ):
    return math.sqrt(math.fabs(count-previousCount)) * (count-previousCount)/float(previousCount)

# Determine hotness
df.loc[:,"Hotness"] = df.apply(lambda row: hotness(row['Count'],row['m1Count']), axis=1)

# Get Count, Hotness from the future year
df.loc[:,'FutureCount'] = df.groupby(['Name','Gender'])['Count'].shift(-1)
df.loc[:,'FutureRank'] = df.groupby(['Name','Gender'])['Rank'].shift(-1)
df.loc[:,'FutureHotness'] = df.groupby(['Name','Gender'])['Hotness'].shift(-1)


# Remove rows with missing data
df = df[ ~pd.isnull(df["m4Rank"]) ]


# Remove years >= 2012 from test and train and other criteria
df_red = df[ (df['Year'] < firstYearInValidation) & ((df['Rank']<=1000) | (df['FutureRank']<=1000)) & ~pd.isnull(df["FutureCount"]) ]


# Get all the columns from the dataframe.
trainvar_columns = df.columns.tolist()
# Filter the columns to be used for training.
trainvar_columns = ["Count","Rank","m1Count","m1Rank","m2Count","m2Rank","m3Count","m3Rank","m4Count","m4Rank"]

# Store the variable we'll be predicting on.
target_hotness = "FutureHotness"
target_count = "FutureCount"


# Generate the training set.  Set random_state to be able to replicate results.
train = df_red.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = df_red.loc[~df_red.index.isin(train.index)]


# Predict future hotness

# Initialize the model with some parameters.
model_hotness = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model_hotness.fit(train[trainvar_columns], train[target_hotness])
# Make predictions on test set.
#predictions = model.predict(test[columns])
predictions_hotness = model_hotness.predict(df[trainvar_columns])

## Predict future count
# Initialize the model with some parameters.
model_count = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model_count.fit(train[trainvar_columns], train[target_count])
# Make predictions on test set.
# Use for validation of the test
#predictions = model.predict(test[columns])
predictions_count = model_count.predict(df[trainvar_columns])


df.loc[:,"PredFutureHotness"] = predictions_hotness
df.loc[:,"PredFutureCount"] = predictions_count

# Determine predicted hotness
df.loc[:,"PredFutureCountHotness"] = df.apply(lambda row: hotness(row['PredFutureCount'],row['Count']), axis=1)

# To be used later for prediction
df_full = df

# For now, remove low ranked names for training
df = df[ ((df['Rank']<=1000) | (df['FutureRank']<=1000)) ]

# Select subset of columns
df_res = df[ ["Year", "Name", "Gender", "FutureHotness"] ]
df_pred_hotness = df[ ["Year", "Name", "Gender", "PredFutureHotness"] ]
df_pred_count = df[ ["Year", "Name", "Gender", "PredFutureCountHotness"] ]

# Rename true and predictions column
df_res.columns = ["Year", "Name", "Sex", "FutureHotness"]
df_pred_hotness.columns = ["Year", "Name", "Sex", "FutureHotness"]
df_pred_count.columns = ["Year", "Name", "Sex", "FutureHotness"]

# Rank true and predicted hotness for merging 
df_res.loc[:,'FutHotRank'] = df_res.groupby(['Year'])['FutureHotness'].rank(ascending=False)
df_pred_hotness.loc[:,'FutHotRank'] = df_pred_hotness.groupby(['Year'])['FutureHotness'].rank(ascending=False)
df_pred_count.loc[:,'FutHotRank'] = df_pred_count.groupby(['Year'])['FutureHotness'].rank(ascending=False)

# Merge on hotness rank, merge predictions first, then predictions to true
merge_pred = pd.merge(df_pred_hotness,df_pred_count,on=["Year","FutHotRank"],suffixes=('H', 'C'))
merged = pd.merge(df_res,merge_pred,on=["Year","FutHotRank"],suffixes=('Tr', 'Pr'))


# Print results for given year
def printMergedResults( year ):
    if not ( merged["Year"]==year ).any():
        return
    use_df = merged[ merged["Year"]==year ]
    
    sorted = use_df.sort_values(['Year','FutHotRank'],ascending=[0,1])

    print
    print
    print "-------------------------- %d : TOP 10 RISERS -------------------------" % ( year )
    print
    print sorted[:20].reset_index(drop=True)
    print
    print "-------------------------- %d : TOP 10 FALLERS -------------------------" % ( year )
    print
    print sorted[-20:].sort_values(['Year','FutHotRank'],ascending=[0,0]).reset_index(drop=True)
    print



years = set( df[ df["Year"]>=firstYearInValidation ]["Year"] )

for year in years:
    printMergedResults( year )


showColumns = [ "Name", "Year", "Gender", "Count", "PredFutureCount", "PredFutureHotness", "PredFutureCountHotness" ]

def printPrediction( year, useCount ):
    useYear = year-1
    use_df = df_full[ df_full["Year"]==useYear ]

    use_df = use_df[ (use_df["Count"]>30) & ((use_df["Count"]>360) | (use_df["PredFutureCount"]>360)) ]
    if useCount:
        useMethod = "PredFutureCountHotness"
    else :
        useMethod = "PredFutureHotness"

    sorted = use_df.sort_values(['Year',useMethod],ascending=[0,0])

    print
    print
    print "         ======== %d prediction using method: %s ======== " % ( year, useMethod )
    print "-------------------------- Predicted %d : TOP 10 RISERS -------------------------" % ( year )
    print
    print sorted[showColumns][:20].reset_index(drop=True)
    print
    print "-------------------------- Predicted %d : TOP 10 FALLERS -------------------------" % ( year )
    print
    print sorted[showColumns][-20:].sort_values(['Year',useMethod],ascending=[0,1]).reset_index(drop=True)
    print


printPrediction( 2013, True )

printPrediction( 2014, True )

printPrediction( 2015, True )
