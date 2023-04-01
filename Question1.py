# First, let's load the dataset and check its shape and head:
import pandas as pd

url = 'https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv'
df = pd.read_csv(url)

print(df.shape)
print(df.head())

# Let's check the data types and missing values:

print(df.dtypes)
print(df.isnull().sum())


# Let's check the summary statistics:

print(df.describe())

