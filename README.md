# Cluster-Project
Clustering Project for Codeup by Jennifer Lewis and Peter Chavez.
 Creating a prediction model based on Zillow data from 2017 for Single Family properties using clustering techniques.

## <a name="project_description"></a>Project Description:
The purpose of this project is to aquire tax data from the year 2017 from a Zillow database, create clusters to train a model to predict the logerror on future house sales of Single Family properties, and utilize that model to make predictions from a group of unknown house sales.

Goals: 
 - Create a model that can predict the logerror with greater accuracy than baseline.


## <a name="planning"></a>Project Planning: 


### Project Outline:
- Acquire, clean, and prepare the 2017 Zillow data from the database.
- Explore the data to determine what features could be usefull for clustering and modeling.
- Establish a baseline root mean square error to compare to later, split the data appropriately, and then train three different regression models.
- Evaluate the models on the train and validate datasets.
- Use the single best model on the test data to evaluate its performance.
- Create a .csv file with the logerror and the model's predictions for each observation in the test data.
- Include conclusions, takeaways, recommendations, and next steps in the final report.

## <a name="dictionary"></a>Data Dictionary  

| Model Target Feature | Definition | Data Type |
| ----- | ----- | ----- |
| logerror |  | float |


---
| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| fips | A code used to identify the region the sale of the property occured down to the county. | |
| | https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt | int |
| parcelid | A unique code used to identify the specific property in the transaction. | int |
| bedrooms | The number of bedrooms in the home. | float |
| bathrooms | The number of bathrooms in the home. | float |
| area | The total square footage of the property. | float |
| acres | The total lot size scaled by 43560 sq ft/acre. | float |
| age | The year that the home was sold compared to when it was originally built. | int |
| taxamount | The amount of taxes piad yearly on the property as of 2017. | float |
| LA | The specified county according to the FIPS number 6037. | int |
| Orange | The specified county according to the FIPS number 6059. | int |
| Ventura | The specified county according to the FIPS number 6111. | int |
| structure_dollar_per_sqft | The cost per square foot of the property. | float |
| bath_bed_ratio | The ratio of bathrooms to bedrooms. This helps to compare the use of the sq ft of a property. | float |


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



### Exploring the remaining columns.





### SelectKBest and RFE



### Data Exploration Takeaways:



## <a name="model"></a>Modeling:

#### Clustering



#### Training Stats
| Model | rmse | r2_score |
| ---- | ---- | ---- |
| Baseline | .436  | - |
| OLS | 195242.06 | 0.190 |  
| LassoLars | 195240.73 | 0.190 |  
| GLM | 195075.16 | 0.191 |  

- Which model was chosen?

## Testing

- 

#### Testing Stats
| Model | rmse | r2_score |
| ---- | ---- | ---- |
| Baseline | .436  | - |
| OLS_test | 138057.02 | 0.436  |  

## <a name="conclusion"></a>Conclusions:

- 

## <a name="next_steps"></a>Next steps:

- 
- 
