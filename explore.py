import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import urllib.request
from PIL import Image
from pydataset import data
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
from wrangle import *
import env
import os

pd.options.mode.chained_assignment = None

def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedroomcnt', 'bathroomcnt', 'cal_fin_sqf', 'tax_val', 'taxamount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()

def get_hist(df):
    ''' Gets histographs of acquired continuous variables.'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()

def get_box_hist_viz(df):
    get_hist(df)
    get_box(df)

def correlation_viz(df, features):
    
    # make correlation plot
    df_corr = df.copy()
    df_corr = df[features].corr()
    plt.figure(figsize=(16,9))
    sns.heatmap(df_corr, annot = True, mask= np.triu(df_corr), linewidth=.65, cmap='Spectral')
    plt.show()

def target_split(df, target):
   
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_df = df.drop(columns=[target])
    y_df = df[[target]]

    return X_df, y_df


def select_kbest(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    
    return: a list of the selected features from the SelectKBest process
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    cols = kbest.get_support(indices=True)
    return X.columns[cols]
  
   

def rfe(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    
    return: a list of the selected features from the recursive feature elimination process
    '''
    rf = RFE(LinearRegression(), n_features_to_select=k)
    rf.fit(X, y)
    mask = rf.get_support()
    return X.columns[mask]


def hist_plot(df):
    '''
    Plots Histograms for columns in the input Data Frame, 
    '''
    plt.figure(figsize=(26, 6))

    cols = [col for col in df.columns if col not in ['parcelid', 'yearbuilt', "lot_dollar_sqft_bin", 'lotsizesquarefeet', 'fips', 'age', 'age_bin', 'date', 'LA', 'Orange', 'Ventura', 'cola', 'longitude', 'latitude', 'regionalidcity', 'regionalidzip']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1 <-- Good to note
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        df[col].hist(bins=5)
        # We're looking for shape not actual details, so these two are set to 'off'
        plt.grid(False)
        plt.ticklabel_format(useOffset=False)
        # mitigate overlap: This is handy. Thank you.
        plt.tight_layout()

    plt.show()


def box_plot(df, cols = ['bathrooms', 'bedrooms', 'sqft_bin', 'fips','acres_bin', 'structure_dollar_sqft_bin', 'bath_bed_ratio']
):
    ''' 
    Takes in a Data Frame, and list of columns
    Plots Boxplots of input columns.
    '''
    
    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        sns.boxplot(data=df[[col]])
        plt.grid(False)
        plt.tight_layout()

    plt.show()


def plot_variable_pairs(df):
    '''
    Takes in DF (Train Please,) and numerical column pair 
    Returns the plotted out variable pairs heatmap and numerical pairplot.
    '''
    pair1 = ['area', 'logerror']
    pair2 = ['acres', 'logerror']
    df_corr = df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(df_corr, cmap='Purples', annot = True, mask= np.triu(df_corr), linewidth=.5)
    plt.title("Correlations between variables")
    plt.show()
    sns.pairplot(df[pair1].sample(1_000), corner=True, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.show()
    sns.pairplot(df[pair2].sample(1_000), corner=True, kind='reg', plot_kws={'line_kws': {'color':'red'}})
    plt.show()

def plot_categorical_and_continuous_vars(df, cat, target = 'logerror'):
    '''
    Takes a Data Frame,
    plots a continuous varible [default target value of logerror](please enter an int/float column) as y
    sorted by categorical variable as x.
    Returns swarm plot, violin plot, and cat plot for each.
    '''

    fig, ax = plt.subplots(figsize=(16, 6))
    for col in cat:
        
        sns.swarmplot(data=df.sample(800), x=col, y=target, s=3)
        # fixing number of tick marks
        plt.locator_params(axis='x', nbins=10)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.show()

        sns.violinplot(data=df.sample(1_000), x=col, y=target, s=4)
         # fixing number of tick marks
        plt.locator_params(axis='x', nbins=10)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.show()
        
        sns.catplot(data=df.sample(500), x=col, y=target, s=4)
         # fixing number of tick marks
        plt.locator_params(axis='x', nbins=10)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.show()


def scale(df, columns_for_scaling = ['bathrooms', 'bedrooms', 'area', 'age', 'acres', 'bath_bed_ratio', 'sqft_bin', 'structure_dollar_per_sqft'], scaler = MinMaxScaler()):
    '''
    Takes in df, columns to be scaled (default: bedrooms, bathrooms, tax_value), 
    and scaler (default: MinMaxScaler(); others can be used ie: StandardScaler(), RobustScaler(), QuantileTransformer())
    returns a copy of the df, scaled.
    '''
    scaled_df = df.copy()
    scaled_df[columns_for_scaling] = scaler.fit_transform(df[columns_for_scaling])
    return scaled_df


def elbow(train_scaled, variables, start = 2, finish = 15):
    X = train_scaled[variables]
    pd.Series(
        {k: KMeans(k).fit(X).inertia_
        for k in range(start,finish)}).plot(marker='*')
    plt.ylabel('Inertia')
    plt.xlabel("k Number of clusters")
    plt.title("Change in Inertia as K Increases")
    plt.show()


def cluster(df_scaled, variables, n_clusters = 5):
    '''
    Takes Scaled Train Dataframe, variables to use to find clusters, and number of clusters desired [default: 5]
    Fits them with KMeans,
    Returns Cluster Predictions [Save as a column in your dataframe.]
    '''
    X = df_scaled[variables]
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    return kmeans.predict(X)


def nice_hist(df):
    df_plot = df.copy()
    df_plot1 = df_plot.iloc[ : ,:15]
    hist_plot(df_plot1)
    plt.show()
    df_plot2 = df_plot.iloc[ : ,15:]
    hist_plot(df_plot2)
    plt.show()

def drop_loc(df):
    df = df.drop(columns=["cola", "lot_dollar_sqft_bin", "land_dollar_per_sqft", "yearbuilt", 'regionidcity','regionidzip','structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'longitude', 'latitude'])
    return df


def log_error_mean(df):
    cols = ['bathrooms', 'bedrooms', 'sqft_bin', 'fips','acres_bin', 'structure_dollar_sqft_bin', 'bath_bed_ratio']
    for col in cols:
        print(f'======{col}======')
        print(df.logerror.groupby(df[col]).agg(['mean']).T)
        print("  ")


def fips_relations(train):
    sns.relplot(data = train, x = "area", y= "tax_value", hue = "fips")
    plt.show()
    sns.relplot(data = train, x = "acres", y= "tax_value", hue = "fips")
    plt.show()
    sns.relplot(data = train, x = "area", y= "structure_dollar_per_sqft", hue = "fips")
    plt.show()


def plot_df(train):
    plt_df = train[['bathrooms', 'bedrooms', 'area', 'age',
                        'acres', 'bath_bed_ratio', 'sqft_bin', 'structure_dollar_per_sqft', 'fips']]

    sns.pairplot(data=plt_df, hue='fips', corner = True)
    plt.show()