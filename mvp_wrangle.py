import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import env
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans


import warnings
warnings.filterwarnings('ignore')


# ============ Data Acquisition =============

def get_connection(db, user = env.username, host = env.hostname, password = env.password):
    '''
    Takes database name for input,
    returns url, using user, password, and host pulled from your .env file.
    PLEASE save it as a variable, and do NOT just print your credientials to your document.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_data():
    '''
    This Function Returns the Zillow 2017 Data Frame. If zillow.csv is in the local
    folder it will pull and use that. Otherwise, it will query the SQL server, pulling
    the queried data from properties_2017 as well as logerror from prediction_2017
    for all properties with the propertylandusetypeid '261', Single Family Residential.
    Then saves the data as a .csv
    '''
    filename= 'zillow.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename, header=0, delim_whitespace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns.values[5] = "fips"

        return df
    else:
        query = '''
        select prop.parcelid
            , pred.logerror
            , bathroomcnt
            , bedroomcnt
            , calculatedfinishedsquarefeet
            , fips
            , latitude
            , longitude
            , lotsizesquarefeet
            , regionidcity
            , regionidcounty
            , regionidzip
            , yearbuilt
            , structuretaxvaluedollarcnt
            , taxvaluedollarcnt
            , landtaxvaluedollarcnt
            , taxamount
        from properties_2017 prop
        inner join predictions_2017 pred on prop.parcelid = pred.parcelid
        where propertylandusetypeid = 261;
        '''
        df = pd.read_sql(query, get_connection('zillow'))
        df.to_csv(filename)

        return df

def get_counties(df):
    '''
    NEW DOCSTRING NEEDED
    OLD ONE:
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies

def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                       )

    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet


    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )


    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                      )


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    return df


def remove_outliers(df):
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''

    return df[((df.bathroomcnt <= 7) & (df.bedroomcnt <= 7) & 
               (df.regionidzip < 100000) & 
               (df.bathroomcnt > 0) & 
               (df.bedroomcnt > 0) & 
               (df.acres < 20) &
               (df.calculatedfinishedsquarefeet < 10000) & 
               (df.taxrate < 10)
              )]


def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=13)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=13)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return partitions


def wrangle_zillow():
    train, X_train, X_validate, X_test, y_train, y_validate, y_test = split(remove_outliers(create_features(get_counties(get_zillow_data()))), 'logerror')
    
    return train, X_train, X_validate, X_test, y_train, y_validate, y_test


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


def cluster(train_scaled, variables, n_clusters = 5):
    '''
    Takes Scaled Train Dataframe, variables to use to find clusters, and number of clusters desired [default: 5]
    Fits them with KMeans,
    Returns Cluster Predictions [Save as a column in your dataframe.]
    '''
    X = train_scaled[variables]
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    return kmeans.predict(X)


# ========= Visualize ===========

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


def box_plot(df, cols):
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


def plot_variable_pairs(df, num_pair):
    '''
    Takes in DF (Train Please,) and numerical column pair 
    Returns the plotted out variable pairs heatmap and numerical pairplot.
    '''
    df_corr = df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(df_corr, cmap='Purples', annot = True, mask= np.triu(df_corr), linewidth=.5)
    plt.title("Correlations between variables")
    plt.show()
    
    sns.pairplot(df[num_pair].sample(1_000), corner=True, kind='reg', plot_kws={'line_kws':{'color':'red'}})
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