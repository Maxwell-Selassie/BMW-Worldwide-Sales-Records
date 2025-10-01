# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# %% [markdown]
# # ***Load dataset and perform basic exploratory data analysis***

# %%
# load dataset
if os.path.exists('../data/BMW sales data.csv'):
    df = pd.read_csv('../data/BMW sales data.csv')
else:
    raise FileNotFoundError('File not found! Please check file path and try again!')


# %%
df.head()

# %%
df.tail()

# %%
# overview of the dataset
df.shape # shape of dataset

# %%
df.info() # returns non-null counts and data types

# %%
# a short descriptive summary of the dataset
df.describe(include='all').round(2).T

# %% [markdown]
# # ***Comprehensive exploratory data analysis***

# %%
missing_values = df.isnull().sum()
missing_pct = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({
    'missing values' : missing_values,
    'missing percentage' : missing_pct.round()
})
print(missing_data[missing_data['missing values'] > 0])

# %% [markdown]
# `No missing data`

# %%
df[df.duplicated()] 

# %% [markdown]
# `No duplicated data`

# %%
df.columns.tolist() # names of the features in the dataset

# %%
# basic numeric features analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('='*50)
print('BASIC NUMERIC COLUMNS ANALYSIS')
print('='*50,'\n')
print(numeric_cols,'\n')
for i,col in enumerate(numeric_cols,1):
    print(f'{i}. {col:<13} : Min_value : {df[col].min():<12} Max_value : {df[col].max()}')

# %%
# basic categorical feature analysis
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print('='*50)
print('BASIC CATEGORICAL COLUMNS ANALYSIS')
print('='*50,'\n')
for i,col in enumerate(categorical_cols,1):
    print(f'{i}. {col:<20} : Unique features : {df[col].unique()}')

# %%
# outlier detection using IQR
def outlier_detection(df,col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3 - q1 # interquartile range

    lower_bound = q1 - 1.5 * IQR 
    upper_bound = q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers, lower_bound, upper_bound

for i,col in enumerate(numeric_cols,1):
    outliers, lower_bound, upper_bound = outlier_detection(df,col)
    print(f'{i}. {col:<13} : {len(outliers)} outliers   (Range: {lower_bound:.2f} - {upper_bound:.2f} )')

# %%
# data distribution analysis 
from scipy.stats import normaltest
for i,col in enumerate(numeric_cols,1):
    stat, p_value = normaltest(df[col])
    is_normal = 'Normal' if p_value > 0.05 else 'Not Normal'
    print(f'{i}. {col:<13} : P-value = {p_value} ({is_normal})')

# %%
# correlation matrix
plt.figure(figsize=(12,5))
corr = df[numeric_cols].corr(method='spearman')
sns.heatmap(data=corr, annot= True, alpha= 0.7, center= 0, fmt= '.2f',cmap='Blues')
plt.title('Correlation Heatmap',fontsize= 12, fontweight = 'bold')
plt.grid(True, alpha = 0.3)
plt.show()

# %% [markdown]
# # ***INSIGHTS GENERATED FROM DATA***
# * `File` : `BMW sales data.csv`
# * `Shape` : `50,000 rows, 11 features`
# * `Goal` : `Predict BMW car prices from sales data from the year 2010 t0 2024`
# 
# # *Feature Table*
# |Feature       | Type        | Role          | Description   |
# |---------------|-------------|---------------|---------------|
# |`Year`       | int          | Feature  | Manufacturing year of the car |
# |`Engine Size L` | float | Feature | Engine displacement (in litres) |
# |`Mileage KM`  | int | Feature | Total distance driven by the car (in kilometres) |
# |`Sales Volume` | int |Feature |Number of BMW units sold|
# |`Model`  | object | Feature | Car Model (e.g. 5 Series, X5, M3) |
# |`Region`     | object | Feature | Regions in which the car was sold (e.g. Asia, Africa, Middle East, Europe) |
# |`Color` | object       | Feature  | Color of the car (e.g. red, blue, silver) |
# |`Fuel Type` | object | Feature | Fuel type the car uses (e.g. Electric, hybrid, diesel) |
# |`Transmission`  | object | Feature  | Whether the car is automatic or manual |
# |`Sales Classification`   |object | Feature | Whether the sale is considered a high or low sale |
# |`Price (USD)`  | object | `Target`  | Sale Price of the BMW car |

# %% [markdown]
# # ***Advanced Visualizations***
# 
# ***`Univariate Analysis`***

# %%
plt.figure(figsize=(18,20))

# Distribution Plots
for i, col in enumerate(numeric_cols):
    plt.subplot(5, 3, i+1)
    sns.histplot(data=df, x=col, kde=True, color= 'green')
    plt.title(f'Distribution of {col.title()}', fontsize = 11, fontweight = 'bold')
    plt.ylabel('Frequency')

# boxplots for outlier detection
for i,col in enumerate(numeric_cols[1:]):
    plt.subplot(5, 3, 6+i)
    sns.boxplot(data=df, y= col, color= 'indigo', saturation=0.4)
    plt.title(f'Box Plot - {col}', fontsize = 12, fontweight = 'bold')

plt.subplot(5, 3 ,10)
plt.pie(data=df,x=df['Transmission'].value_counts(), labels=df['Transmission'].unique(),
        colors=['green','gold'],explode=[0.03,0.03],shadow=True, autopct='%1.1f%%',startangle=90)
plt.title('Transmission',fontsize=12, fontweight='bold')

plt.subplot(5, 3 ,11)
plt.pie(data=df,x=df['Sales_Classification'].value_counts(), labels=df['Sales_Classification'].unique(),
        colors=['orange','red'],explode=[0.03,0.03],shadow=True, autopct='%1.1f%%',startangle=45)
plt.title('Sales Classification',fontsize=12, fontweight='bold')
plt.grid(True, alpha= 0.3)

plt.subplot(5, 3, 12)
ax = sns.countplot(data=df, x= df['Fuel_Type'],color='maroon', alpha=0.6,width=0.5,gap=0.2)
for container in ax.containers:
    ax.bar_label(container,label_type='edge')
plt.title('Fuel Type Distribution',fontsize= 12, fontweight= 'bold')
plt.ylabel('Frequency')

plt.subplot(5, 3, 13)
ax = sns.countplot(data=df, x= df['Model'],color='maroon', alpha=0.6,width=0.5,gap=0.2)
for container in ax.containers:
    ax.bar_label(container,label_type='edge')
plt.title('Model Distribution',fontsize= 12, fontweight= 'bold')
plt.ylabel('Frequency')

plt.subplot(5, 3, 14)
ax = sns.countplot(data=df, x= df['Region'],color='maroon', alpha=0.6,width=0.5,gap=0.2)
for container in ax.containers:
    ax.bar_label(container,label_type='edge')
plt.title('Region Distribution',fontsize= 12, fontweight= 'bold')
plt.ylabel('Frequency')


plt.tight_layout()
plt.show()

# %% [markdown]
# ***`Bivariate Analysis`***

# %%
plt.figure(figsize=(20,18))

# relationship with years and sales over time
plt.subplot(3, 3, 1)
sns.lineplot(data=df, x= 'Year', y= 'Price_USD',markers= 'x')
plt.title('Prices Over Years')
plt.grid(True,alpha=0.3)

# scatter plots with regression lines
plt.subplot(3, 3, 2)
sns.scatterplot(data=df,x='Mileage_KM',y='Price_USD',alpha=0.6,color='purple',markers='x')
sns.regplot(data=df,x='Mileage_KM',y='Price_USD',scatter=False,color='red')
plt.title('Mileage(KM) vs Price(USD)',fontsize=12, fontweight= 'bold')

plt.subplot(3, 3, 3)
sns.scatterplot(data=df,x='Engine_Size_L',y='Price_USD',alpha=0.6,color='purple',markers='x')
sns.regplot(data=df,x='Engine_Size_L',y='Price_USD',scatter=False,color='red')
plt.title('Engine_Size_L vs Price(USD)',fontsize=12, fontweight= 'bold')

plt.subplot(3, 3, 4)
sns.scatterplot(data=df,x='Sales_Volume',y='Price_USD',alpha=0.6,color='purple',markers='x')
sns.regplot(data=df,x='Sales_Volume',y='Price_USD',scatter=False,color='red')
plt.title('Sales Volume vs Price(USD)',fontsize=12, fontweight= 'bold')
plt.ylabel('Average Price (USD)')

# sales per region
plt.subplot(3, 3, 5)
df.groupby('Region')['Price_USD'].mean().sort_values(ascending=False).plot.bar(
    color='purple', alpha= 0.7)
plt.title('Average Sales Per Region',fontsize=12, fontweight = 'bold')
plt.ylabel('Average Price (USD)')

# sales per fuel type
plt.subplot(3, 3, 6)
df.groupby('Fuel_Type')['Price_USD'].mean().sort_values(ascending=False).plot.bar(
    color='indigo', alpha=0.7
)
plt.title('Average Sales Per Fuel Type',fontsize=12, fontweight = 'bold')
plt.ylabel('Average Price (USD)')

# sales per transmission
plt.subplot(3, 3, 7)
df.groupby('Transmission')['Price_USD'].mean().sort_values(ascending=False).plot.bar(
    color='green', alpha=0.7)
plt.title('Average Sales Per Transmission',fontsize=12, fontweight = 'bold')
plt.ylabel('Average Price (USD)')

# sales per model
plt.subplot(3, 3, 8)
df.groupby('Model')['Price_USD'].mean().sort_values(ascending=False).plot.bar(
    color='red', alpha=0.6)
plt.title('Average Sales Per Model',fontsize=12, fontweight = 'bold')
plt.ylabel('Average Price (USD)')

# sales per color
plt.subplot(3, 3, 9)
df.groupby('Color')['Price_USD'].mean().sort_values(ascending=False).plot.bar(
    color='indigo', alpha=0.7)
plt.title('Average Sales Per Color',fontsize=12, fontweight = 'bold')
plt.ylabel('Average Price (USD)')

plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # ***ADVANCED FEATURE ENGINEERING***

# %%
df['Car_Age'] = df['Year'].apply(lambda x: df['Year'].max() - x)

# %% [markdown]
# # ***Comprehensive Statistical Analysis***

# %%
# feature correlation with target
num_cols = df.select_dtypes(exclude='object').columns
correlation_with_target = df[num_cols].corr(method='spearman')['Price_USD'].drop('Price_USD').sort_values(key=abs,ascending=False)
correlation_with_target

# %%
# ANOVA test for categorical values
from scipy import stats
for i,col in enumerate(categorical_cols,1):
    groups = [group['Price_USD'].values for name, group in df.groupby(col)]
    f_stats, p_value = stats.f_oneway(*groups)
    significance = 'Significant' if p_value < 0.05 else 'Not significant'
    print(f'{i}. {col:<20} : F-Statistics = {f_stats:.3f} , p-value = {p_value:.3f} ({significance})')


