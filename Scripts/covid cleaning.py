import pandas as pd
import numpy as np
import re


# load the data
df = pd.read_csv("data.csv")


# drop first row and reset index
df.drop(0, inplace=True)


# drop columns
df.drop(columns=["Active Cases", "Tests"], inplace=True)


# remove comma from strings
df['Cases'] = df['Cases'].str.replace(",", '').str.strip()
df['Deaths'] = df["Deaths"].str.replace(",", '').str.strip()
df['Recoveries'] = df['Recoveries'].str.replace(",", '').str.strip()

df['Cases'] = df['Cases'].replace('–', np.nan)



# fix data type
column = ['Cases', 'Deaths', 'Recoveries']


for col in column:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Date'] = df['Date'].str.strip()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')



def clean_province_name(name):
    if pd.isna(name):
        return name
    name = str(name)

    # Remove unwanted characters (e.g., \x, \n, etc.)
    name = re.sub(r'[^a-zA-Z\s\-]', '', name)

    # Remove terms like "province", case-insensitive
    name = re.sub(r'\bprovince\b', '', name, flags=re.IGNORECASE)

    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)

    return name.strip().title()  # Capitalize properly


df['Province'] = df['Province'].apply(clean_province_name)


province_corrections = {'Jawzjon': 'Jawzjan',
                       'Herat': 'Hirat',
                       'Sar-E-Pul': 'Sar-E Pol',
                       'Jowzjan': 'Jawzjan',
                      'Nimruz': 'Nimroz' ,
                       'Nooristan': 'Nuristan',
                       'Panjshir': 'Panjsher',
                       'Paktia': 'Paktya',
                       'Daykundi': 'Dykundi'}


# Apply corrections using map and fallback to original if not in dict
df['Province'] = df['Province'].apply(lambda x: province_corrections.get(x, x))


# fill the missing values
columns_to_convert = ["Cases", "Deaths", "Recoveries"]
for col in columns_to_convert:
    df[col] = df.groupby("Province")[col].transform(lambda x: x.fillna(x.median()))


# remove outliers
for col in ['Cases', 'Deaths', 'Recoveries']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    final = df[(df[col] >= lower) & (df[col] <= upper)]


#clean column names
df.columns = df.columns.str.lower()
df.sort_values(by='date', inplace=True)


# save the data
df.to_csv("cleaned-data.csv", index=False)