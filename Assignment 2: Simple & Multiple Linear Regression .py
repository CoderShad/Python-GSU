import pandas as pd
import csv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as seabornInstance
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import yfinance as yf
from scipy import stats
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
fred = Fred(api_key='c6a3529843d2d1e6ff7ea9b09fdb27ea')


############ Part 1 - Done By Shad Sabbir ################


### i. Download SPY ETF which tracks S&P500 index
SPY = yf.download("SPY", start = '2008-01-01', end = '2023-12-31')
GOOGL = yf.download("GOOGL", start = '2008-01-01', end = '2023-12-31')

### ii. Takes only the last adj closing price of each month and calculates monthly returns
SPY_Monthly_Return = (SPY['Adj Close'].resample("M").last().pct_change().dropna())
GOOGL_Monthly_Return = (GOOGL['Adj Close'].resample("M").last().pct_change().dropna())
Stocks = pd.DataFrame({'SPY': SPY_Monthly_Return, 'GOOGL': GOOGL_Monthly_Return })
print(SPY_Monthly_Return.head())
print(GOOGL_Monthly_Return.head())

# iii. Plot a histogram of SPY's monthly returns using only matplotlib
plt.figure(figsize=(10, 6))
plt.hist(SPY_Monthly_Return, bins=20, edgecolor='black')
# Add titles and labels
plt.title('Histogram of SPY Monthly Returns (2008 - 2023)')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
# Show the plot
plt.show()
# Plot a histogram of Google's monthly returns using only matplotlib
plt.figure(figsize=(10, 6))
plt.hist(GOOGL_Monthly_Return, bins=20, edgecolor='black')
# Add titles and labels
plt.title('Histogram of GOOGL Monthly Returns (2008 - 2023)')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
# Show the plot
plt.show()
# Print Summary Statistic - displayed in percentages for easier interpretation
print(SPY_Monthly_Return.describe()*100)
print(GOOGL_Monthly_Return.describe()*100)
#Correlation Matrix
print(Stocks.corr())


### iv. Get 3-month Treasury Bill (risk-free rate)
tbill = fred.get_series('TB3MS', start='2008-01-01', end='2023-12-31').resample("M").last().dropna()
print(tbill.tail())
# Align tbill with stock returns (convert tbill to monthly percentages for excess returns)
tbill = tbill / 100 / 12  # Convert annualized rate to monthly risk-free rate
print(tbill.tail())

#### v. Align data in a dataframe and matrix of return series
data = pd.DataFrame({'GOOGL_Return': GOOGL_Monthly_Return, 'SPY_Return': SPY_Monthly_Return, 'TBILL': tbill}).dropna()
data_matrix = data.to_numpy()

### vi. Fnding Beta using CAPM model
# Calculate excess returns
data['Excess_GOOGL'] = data['GOOGL_Return'] - data['TBILL']
data['Excess_SPY'] = data['SPY_Return'] - data['TBILL']
# Perform regression (CAPM Model)
X = sm.add_constant(data['Excess_SPY'])  # Add constant for intercept
model = sm.OLS(data['Excess_GOOGL'], X).fit()
# Print the regression results
print(model.summary())
# vii. Extract Beta from the model (slope of the regression)
beta = model.params['Excess_SPY']
print(f"Beta for GOOGL: {beta}")

### vii.  sTest Null Hypothesis
print(""" According to the regression summary
Google's Beta or Coefficient is 1.0514 according to the slope of the regression.
Since P-value is less than 0.05, we reject the null hypothesis and
conclude that GOOGL’s returns are significantly related to the market (SPY),
meaning beta is statistically significant and is a nonzero number.
""")

### viii. Estimate beta using beta formula
#Calulate Beta Alternative way
print(Stocks.cov())
Cov = Stocks.cov()
print(SPY_Monthly_Return.var())
Var = SPY_Monthly_Return.var()
beta2 = (Cov/Var)

# Google's beta is 1.050371 based on covariance of the stock return with market return and variance of market return

### ix. Interpretation of the coefficient estimate
print("""
Interpretation:

The beta coefficient in the regression output represents the sensitivity of GOOGL's excess returns to SPY's excess returns (which represents the market in this case).
There is a positive relationship between SPY (market) and GOOGL returns
Specifically, for every 1% increase in the excess return of SPY, GOOGL's excess return is expected to increase by about 1.05%
In other words, GOOGL exhibits slightly higher market risk compared to SPY.
""")

### x. Model Accuracy Comments
print("""
Model Accuracy Comments: 
The model explains 40% of GOOGL's excess returns, with a statistically significant relationship between GOOGL and SPY. 
However, 60% of the variation in GOOGL's returns remains unexplained, suggesting that other factors 
should be considered to improve the model's accuracy. The small standard errors indicate 
that the coefficient estimates are precise, particularly for the market factor (SPY).
""")

### xi. Scatter plot of Excess_GOOGL vs. Excess_SPY
plt.scatter(data['Excess_SPY'], data['Excess_GOOGL'], color='blue', label='Data Points')
# Generate predicted values based on the regression model
predicted_values = model.predict(X)
# Plot the regression line
plt.plot(data['Excess_SPY'], predicted_values, color='red', label='Fitted Line')
# Add labels and title
plt.xlabel('Excess SPY Returns')
plt.ylabel('Excess GOOGL Returns')
plt.title('Excess GOOGL Returns vs. Excess SPY Returns')
# Show the legend
plt.legend()
# Display the plot
plt.show()

### xii. Discussing alpha as estimated by the fitted model.
print("""
In the regression model, I got a y interecpt of 0.053 which I'm thinking of as alpha. A positive alpha/Y-intercept means 
that the stock is delivering returns above what CAPM would predict based on its risk (beta).

""")

######################################        Part 2  by Deeksha Shukla and Sakshi Bhanushali   #############################################

credit_data = pd.read_csv("/Users/SSUF/Desktop/credit.csv")

### i.
dimensions = credit_data.shape
print("Dimensions of the Credit data:", dimensions)

### ii.
summary_stats = credit_data.describe()
print("Summary Statistics for the Credit data:\n", summary_stats)

### iii.
student_percentage = (credit_data['Student'].value_counts(normalize=True).get('Yes', 0) * 100)
print(f"Percentage of Students: {student_percentage:.2f}%")
#What is the percentage of Female in the Credit data?
female_percentage = (credit_data['Gender'].value_counts(normalize=True).get('Female', 0) * 100)
print(f"Percentage of Female: {female_percentage:.2f}%")
#What is the percentage of Student who are Female in the Credit data?
students = credit_data[(credit_data['Student'] == 'Yes')]
female_percentage = (students['Gender'].value_counts(normalize=True).get('Female', 0) * 100)
print(f"Percentage of Students who are Female: {female_percentage:.2f}%")

####2B####

adv = credit_data
# Create an interaction term between CreditRating and Student
adv['CreditRating_Student'] = adv['Rating'] * (adv['Student'] == 'Yes').astype(int) # Changed 'CreditRating' to 'Rating' assuming that was the intended column
print(adv.head())

### you can use the categorical variable Gender directly in the model using flollwing
result = smf.ols(formula='Balance ~ Student + Rating + CreditRating_Student', data=credit_data).fit()
print(result.summary())

# Define the model formula
formula = 'Balance ~ Rating + Student + CreditRating_Student'
print(formula)

# Fit the model
model = smf.ols(formula, data=adv).fit()
print(model)

# Print the summary of the model
print(model.summary())
######################################        Part 3 by Kihyun Park      #############################################
# Function to load the Credit data
    # Load the Credit data

load_credit_data = credit_data

# Part 3.i: Simple linear regression - Age vs. Credit Card Balance
model_simple = smf.ols('Balance ~ Age', data=credit_data).fit()
print("Simple Linear Regression Summary:")
print(model_simple.summary())

# Part 3.ii: Multiple linear regression - Age and Credit Rating vs. Credit Card Balance
model_multiple = smf.ols('Balance ~ Age + Rating', data=credit_data).fit()
print("\nMultiple Linear Regression Summary:")
print(model_multiple.summary())

# Part 3.iii: Compare effect of Age from part (i) and (ii)
print("\nComparison of Age effect:")
print(f"Age coefficient in simple regression: {model_simple.params['Age']:.4f}")
print(f"Age coefficient in multiple regression: {model_multiple.params['Age']:.4f}")
print("\nExplanation:")
print("The effect of Age on Credit Card Balance differs between the simple and multiple regression models.")
print("In the simple regression, the Age coefficient represents the total effect of Age on Balance.")
print("In the multiple regression, the Age coefficient represents the partial effect of Age on Balance,")
print("controlling for Credit Rating. The difference suggests that Credit Rating explains some of the")
print("variation in Balance that was previously attributed to Age in the simple regression model.")

# Part 3.iv: Create dummy variables based on Age distribution
credit_data['Age_40_and_below'] = (credit_data['Age'] <= 40).astype(int)
credit_data['Age_41_to_56'] = ((credit_data['Age'] > 40) & (credit_data['Age'] <= 56)).astype(int)
credit_data['Age_over_56'] = (credit_data['Age'] > 56).astype(int)

print("\nAge Group Distribution:")
print(credit_data[['Age_40_and_below', 'Age_41_to_56', 'Age_over_56']].sum())

# Visualize Age distribution
plt.figure(figsize=(10, 6))
plt.hist(credit_data['Age'], bins=20, edgecolor='black')
plt.title('Distribution of Age in Credit Data')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.axvline(x=40, color='r', linestyle='--', label='Age 40')
plt.axvline(x=56, color='g', linestyle='--', label='Age 56')
plt.legend()
plt.show()

# Fit the regression model with age dummy variables
model_age_dummy = smf.ols(formula='Balance ~ Age_40_and_below + Age_41_to_56 + Age_over_56', data=credit_data).fit()
print(model_age_dummy.summary())
