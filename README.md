# Cluster-Project
Clustering Project for Codeup by Jennifer Lewis and Peter Chavez.
 Creating a prediction model based on Zillow data from 2017 for Single Family properties using clustering techniques.

## <a name="project_description"></a>Project Description:
The purpose of this project is to aquire tax data from the year 2017 from a Zillow database, create a model to predict the tax value on future house sales of Single Family properties, and utilize that model to make predictions from a group of unknown house sales.

Goals: 
 - Create a model that can predict the tax evaluation with greater accuracy than baseline.


## <a name="planning"></a>Project Planning: 


### Project Outline:
- Acquire, clean, and prepare the 2017 Zillow data from the database.
- Explore the data to determine what features could be usefull for regression modeling.
- Establish a baseline root mean square error to compare to later, split the data appropriately, and then train three different regression models.
- Evaluate the models on the train and validate datasets.
- Use the single best model on the test data to evaluate its performance.
- Create a .csv file with the tax value and the model's predictions for each observation in the test data.
- Include conclusions, takeaways, recommendations, and next steps in the final report.

## <a name="dictionary"></a>Data Dictionary  

| Model Target Feature | Definition | Data Type |
| ----- | ----- | ----- |
| tax_val | The evaluated tax value of the property before it was sold. | float |


---
| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| fips | A code used to identify the region the sale of the property occured down to the county. | |
| | https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt | int |
| parcelid | A unique code used to identify the specific property in the transaction. | int |
| bedroomcnt | The number of bedrooms in the home. | float |
| bathroomcnt | The number of bathrooms in the home. | float |
| cal_fin_sqf | The total square footage of the property. | float |
| year_built | The year that the home was originally built. | int |
| taxamount | The amount of taxes piad yearly on the property as of 2017. | float |
| f6037 | The specified county according to the FIPS number 6037. | int |
| f6059 | The specified county according to the FIPS number 6059. | int |
| f6111 | The specified county according to the FIPS number 6111. | int |


## <a name="wrangle"></a>Data Acquisition and Preparation

A function is used to acquire the data via a SQL query. The data is then prepared by another function in the following way:

- Deletes the id columns that contained redundent information
- Drops unnamed columns accidentally created during .csv creation.
- Removes outlying data that may too heavily skew our training models.
- Converts the datatypes of the columns to usable ones.
- Creates dummy columns for the FIPS values.
- Splits the data into an 80% training set, a 20% validate set, and a 20% testing set with 'tax_val' as the target.



## <a name="explore"></a>Data Exploration:

### Measuring the significance of location.

On first glance in the SQL database, the FIPS values were categorical. After some background research, it was apparent that FIPS are like zip codes, they indicate a certain region down to the counties of each state. According to https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt, the 3 counties in the dataset turned out to be Orange county California, LA county California, and Ventura county California. A multi-variance test was run to prove the significance of the FIPS code as it pertains to the property's location. This turned out to be true, therefore it would be included in the modeling later.

### Exploring the remaining columns.

Box plots and histograms were created to peer into the other features. However, it looked like there might be some trouble with multicollinearity. Certain features were then selected to be left out of the models based on this consideration.

To check if this intuition was correct, a correlation plot was made. It showed that this educated guess was correct, but may lead to further feature engineering in the future.

### SelectKBest and RFE

To double-check on any multicollinearity, several runs of SelectKBest and RFE were run on the features to look for variance in weight. Sure enough, it showed up. This doesn't excluded these features from being useful if further engineered, but this inital run of the project was not about diving too deeply into that. Later modeling revealed this as a good preservation of time and effort.

### Data Exploration Takeaways:

As previously discussed, property value is about location. Also, even though the SelectKBest and RFE feature selections showed varying results, if run multiple times, it is easy to see how bedrooms, bathrooms, and taxamount confused the selection. This helped to prove earlier suspicions. You can see this happen if you refresh the project and run the cells again, you will get different features. Make sure not to run the cells on the same instance, or you will get an error.

## <a name="model"></a>Modeling:

#### Training Stats
| Model | rmse | r2_score |
| ---- | ---- | ---- |
| Baseline | .436  | - |
| OLS | 195242.06 | 0.190 |  
| LassoLars | 195240.73 | 0.190 |  
| GLM | 195075.16 | 0.191 |  

- The OLS model was chosen, but it was hard to justify any of the models.

## Testing

- OLS_test Model

#### Testing Stats
| Model | rmse | r2_score |
| ---- | ---- | ---- |
| Baseline | .436  | - |
| OLS_test | 138057.02 | 0.436  |  

## <a name="conclusion"></a>Conclusions:

- Baseline was not beat! However as discussed earlier, this may be due to how these models do not play well with multicollinearity. They seem to improve marginally as more features are added.

## <a name="next_steps"></a>Next steps:

- Further research is needed into better methods to model data for this type of problem.
- In the future, it would be good to engineer more features to push this model to its limit even though it may not be an optimal way to predict tax value of these properties.
