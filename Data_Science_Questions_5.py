import pandas as pd
car_data = pd.read_csv('.../car_data.csv')

'''
Create dummy variables for the Fuel_Type, Transmission and Seller_Type variables.
'''
def create_dummy_variables(df, var_to_ohe=None):
    '''

    Parameters
    ----------
    df : DataFrame
        Takes as input a DataFrame, which contains information related to cars.
    
    var_to_ohe: list
        Default: None. Pass a list of variables to create dummy variables or one-hot-encode

    Raises
    ------
    Exception
        Raises exception if input variable is not TYPE pandas.DataFrame..

    Returns
    -------
    df : DataFrame
        Returns a DataFrame, which one-hot-encoded or created dummy variables.

    '''
    #Throw exception if input variable is not a DataFrame
    if isinstance(df, pd.DataFrame) == False:
        raise Exception(f"Input variable must be type Pandas DataFrame. Input variable is identified as {type(df)}")
    
    #Create a list of variables to one-hot-encode
    if not var_to_ohe:
        var_to_ohe = ['Fuel_Type', 'Transmission', 'Seller_Type']
    else:
        var_to_ohe = var_to_ohe
    
    for variable_name in var_to_ohe:
        #One-hot-encode the variables using pd.get_dummies()
        ohe = pd.get_dummies(df[variable_name])
        #Join new variables to original DataFrame
        df = df.join(ohe)

    return df

car_data = create_dummy_variables(df=car_data)

'''
Create a new column which captures the age of the car as 'new' or 'old'.
'''
#My assumption is that any car built 5 years ago or sooner is considered as new.  
car_data['car_age_category'] = car_data['Year'].apply(lambda x: 'new' if x >= 2016 else 'old')

'''
 Scale the Kms_Driven, Selling_Price, and Present_Price variables (i.e. ensure the variables have the same scale - thousands or tens. You can choose whichever scale you prefer).
'''
#I've decided to scale the tens to thousands. Selling & present price are the only variables in tens
list_of_variables = ['Selling_Price', 'Present_Price']

for variable in list_of_variables:
    #convert the variables to thousands by multiplying by 1000
    car_data[variable] = car_data[variable].apply(lambda x: x * 1000) 
    
'''
Conduct exploratory analysis for the categorical variables. What are you findings?
'''
import seaborn as sns
import matplotlib.pyplot as plt

#Create a list of categorical variables
cat_var = ['Fuel_Type', 'Seller_Type', 'Transmission','car_age_category']
stats_cat_var = pd.DataFrame(columns=cat_var)
for var in cat_var:
    #calculate the descriptive statistics for the categorical variables
    stats_cat_var[var] = (car_data[var].describe())
    #plot a bar graph
    ax = sns.countplot(x=var, data=car_data)
    plt.show(ax)

'''
80% of the cars are petrol, whereas only 20% are diesel, and less than one percent CNG. The dealer sells
the most amount of cars, at 65%. There are more manual cars (87%) compared to automatic cars (13%). Furthermore,
71% of the cars are considered old. Lastly, the categorical groups are imbalanced.
'''

'''
Conduct exploratory analysis for the continuous variables. Ensure you review each variable by itself, 
and in combination with the other variables to identify insights and trends. What are your findings?
'''
#Create a list of continuous variables
cont_var = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner']
stats_cont_var = pd.DataFrame(columns=cont_var)
cont_variables = pd.DataFrame(columns=cont_var)
for var in cont_var:
    #Calculate the descritive statistics for the continuous variables
    stats_cont_var[var] = (car_data[var].describe())
    cont_variables[var] = car_data[var]

kms_driven_median = car_data[cont_var[-1]].median()

#Plot as scatterplot and seperate by groups
ax = sns.scatterplot(x=car_data['Year'], y=car_data['Selling_Price'], hue=car_data['Fuel_Type'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Kms_Driven'], y=car_data['Selling_Price'], hue=car_data['Fuel_Type'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Present_Price'], y=car_data['Selling_Price'], hue=car_data['Fuel_Type'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Owner'], y=car_data['Selling_Price'], hue=car_data['Fuel_Type'])
plt.show(ax)

ax = sns.scatterplot(x=car_data['Year'], y=car_data['Selling_Price'], hue=car_data['Seller_Type'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Kms_Driven'], y=car_data['Selling_Price'], hue=car_data['Seller_Type'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Present_Price'], y=car_data['Selling_Price'], hue=car_data['Seller_Type'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Owner'], y=car_data['Selling_Price'], hue=car_data['Fuel_Type'])
plt.show(ax)

ax = sns.scatterplot(x=car_data['Year'], y=car_data['Selling_Price'], hue=car_data['Transmission'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Kms_Driven'], y=car_data['Selling_Price'], hue=car_data['Transmission'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Present_Price'], y=car_data['Selling_Price'], hue=car_data['Transmission'])
plt.show(ax)
ax = sns.scatterplot(x=car_data['Owner'], y=car_data['Selling_Price'], hue=car_data['Fuel_Type'])
plt.show(ax)

ax = sns.pairplot(cont_variables)
plt.show(ax)

ax = sns.heatmap(cont_variables.corr(), cmap='hsv', annot=True)
plt.show(ax)
'''
On average a car is sold for $2,967.17 less than the current years model. The average number of kilometers driven
are 36,947.2km, while the median are 32,000km. Selling price and present price have a strong positive coorelation.
Year and kilometers driven are somewhat negatively coorelated, therefore we will choose year for the analysis as
it is more coorelated to the dependent variable. The following results are qualitative interpretations of the figures.
The price of cars sold between 2004 - 2018 have increased per year, however there have been greater increases in 
price of diesel and automatic cars, alongside cars sold by a dealer. Diesel, automatic, and cars sold by a dealer 
with fewer kilometers have been sold for more, compared to other cars. Lastly, there seems to be a linear increase
associated with selling price and present price.    
'''

'''
Regression Model
Based on the insights identified, state a hypothesis which you can test with a regression model.
'''
'''
Hypothesis:
    Selling price increases the newer (years) the car and the higher the price of the current model.
    
Null-Hypothesis:
    Selling price does not change the newer (years) the car and the higher the price of the current model.  
'''

'''
Build a linear regression model based on your hypothesis. Interpret the results.
'''

from statsmodels.formula.api import ols 

results=ols("Selling_Price ~ Year + Present_Price", data=car_data).fit()

print(results.summary())

'''
The linear regression model achieved an r-squared of 0.85, which means that 85% of the variance in selling price
can be explained by year and present price. Furthermore, both year and present price were statistically significant
(p<0.005), which means that we are 95% confident that the coeficient of year will fall between 412 - 567 and present
price will fall between 0.499 - 0.551. Lastly, as year increases by one selling price will increase by 489.7 and when
the present price increases by one the selling price increases by 0.52.    
'''
'''
How could you make your model better? Is there data that you think would be helpful to achieve a higherd R2 value?
'''

'''
First we should verrify all the assumtions. We somewhat evaluated linearity by plotting the data and multicollinearity 
by calculating the correlation between the variables. We could further calculate the variance inflation factor to 
evaluate multicollinearity. We should also asses that the errors come from a Gausian distribution, homoskedasticity, and 
that the errors are independent of eachother. Furthermore, we could test is the model would benefit from additional data.
Additional data such as: make and model of the car, type of car (sedan, suv, van, etc...), and previous accidents. 
'''



















