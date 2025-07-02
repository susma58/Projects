'''
For this project, tools used:
- Beautiful soup: to scrape historical CO₂ emissions and temperature data
- Pandas: for cleaning, merging, and manipulating datasets
- NumPy: for calculations and statistical summaries
- Matplotlib / Seaborn / Plotly: for interactive visualizations
'''

'''
Things we need to do throughout this project:
1. Web scraping with beautifulSoup
Scrape CO₂ emissions (or temperature anomalies) by country from apis
2. Data cleaning and preprocessing
Clean inconsistent headers, remove nuulls, and convert datatypes or reshape as needed
3. Analysis using pandad and NumPy
Calculate annual/global averages; identify top 10 emitting countries over time
Track temperature rise alongside CO₂ emissions; use NumPy for rolling avg, % change, correlation etc.
4. Visualizations
Use Seaborn for heatmaps and correlation plots
Plotly or Matplotlib for line charts, bar graphs, scatter plots
Geopandas for a world map of emissions
5. Conclusion
Which countries are the biggest polluters?
Is there a correlation between CO₂ and temperature rise?
Predict future emissions (use numpy for trendlines)
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO


# setup plotting style
sns.set(style='whitegrid')

# scrape CO₂ emissions data from wikipedia
url = "https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions"
respone = requests.get(url)
soup = BeautifulSoup(respone.text, "html.parser")

# extract table
table = soup.find("table", {"class": "wikitable"})

tables = pd.read_html(str(table), header=1)
df = tables[0]
print(df.columns)

print(df.head())
print(df)




# clean and process the data
# df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
df.columns = [col.strip() for col in df.columns]
df.rename(columns={"Location": "Country", "2023": "2023_CO2_Mt"}, inplace= True)

# keep only necessary columns and clean values
print(df.columns)
df = df[["Country", "2023_CO2_Mt"]]
# df = df[df["2023_CO2_Mt"].apply(lambda x: str(x).replace(',', '', 1).isdigit())]

def is_float(val):
    try:
        float(str(val).replace(',', ''))
        return True
    except:
        return False

df = df[df["2023_CO2_Mt"].apply(is_float)]


df["2023_CO2_Mt"] = df["2023_CO2_Mt"].apply(lambda x: float(str(x).replace(',', '')))

# List of non-country entities to exclude
non_countries = ["World", "International Shipping", "European Union", "International Aviation"]

df = df[~df["Country"].isin(non_countries)]


# add percentage of global emissions
df["% of total"] = (df["2023_CO2_Mt"] / df["2023_CO2_Mt"].sum()) * 100
top10 = df.sort_values("2023_CO2_Mt", ascending=False).head(10)



# visualization

# barplot
plt.figure(figsize = (12, 6))
sns.barplot(x= "Country" , y="2023_CO2_Mt", data=top10, palette="Reds_d")
plt.xticks(rotation=45, fontsize=13, fontweight='bold')
plt.xticks(rotation=45, fontsize=13, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')
plt.title("Top 10 CO2 Emitting Countries (2023)", fontsize=25, fontweight='bold', pad=25)
plt.ylabel("CO2 Emissions (Million Tonnes)", fontsize=16, fontweight='bold', labelpad=15)
plt.xlabel("Country", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Pie chart
plt.figure(figsize = (8, 8))
wedges, texts, autotexts = plt.pie(top10["% of total"], labels = top10["Country"], autopct = "%1.1f%%", startangle = 140,
        colors = sns.color_palette("Reds", 10))

for text in texts:
    text.set_fontweight('bold')
    text.set_fontsize(14)

for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_color('black')

plt.title("Share of Global CO2 Emissions (Top 10 Countries, 2023)", fontweight='bold', fontsize=25)
plt.tight_layout()
plt.show()

# add synthetic temperature data and analyze correlation
np.random.seed(0)
df["Temp_Anomaly"] = np.random.normal(loc=1.0, scale=0.5, size=len(df))
correlation = df["2023_CO2_Mt"].corr(df["Temp_Anomaly"])

# scatter plot
plt.figure(figsize = (10, 6))
sns.scatterplot(data=df, x="2023_CO2_Mt", y="Temp_Anomaly", alpha = 0.6)
plt.title(f"CO2 Emissions vs Temperature Anomaly (r = {correlation: .2f})", fontweight='bold', fontsize=25, pad=20)
plt.xlabel("CO2 Emissions (Million Tonnes)", fontsize=16, fontweight='bold', labelpad=15)
plt.ylabel("Temperature Anomaly (°C)", fontsize=16, fontweight='bold', labelpad=15)
plt.tight_layout()
plt.show()


