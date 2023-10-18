# From Eurostat, get the balance data for the Consumer confidence
#indicator, that is seasonally adjusted, for the EU: for the last
#5 years and for each month of 2022. Print histograms.

'''
- Database: Consumer surveys - monthly data (ei_bsco_m)
- Indicator: Consumer confidence indicator (BS_CSMCI)
- Adjusted: Seasonally adjusted data (SA)
- Unit: Balance (BAL)
- Time selection: Default
- Geo selection: Default
- Unit Code: Default
- Grouped indicators: NO
- Excluded dataset title: NO
- Excluded EU Aggregated: NO
- Number of Decimals: 1
'''

import pandas as pd
import matplotlib.pyplot as plt

# Mapping countries to colors for plotting later
country_colors = {
    'European Union - 27 countries (from 2020)': 'cyan',
    'Belgium': 'green',
    'Bulgaria': 'red',
    'Czechia': 'purple',
    'Denmark': 'orange',
    'Germany': 'blue',
    'Estonia': 'magenta',
    'Ireland': 'brown',
    'Greece': 'pink',
    'Spain': 'lime',
    'France': 'teal',
    'Croatia': 'olive',
    'Italy': 'peru',
    'Cyprus': 'gold',
    'Latvia': 'blueviolet',
    'Lithuania': 'seagreen',
    'Luxembourg': 'steelblue',
    'Hungary': 'sienna',
    'Malta': 'darkorange',
    'Netherlands': 'firebrick',
    'Austria': 'palegreen',
    'Poland': 'navy',
    'Portugal': 'darkviolet',
    'Romania': 'mediumblue',
    'Slovenia': 'tomato',
    'Slovakia': 'saddlebrown',
    'Finland': 'peru',
    'Sweden': 'mediumseagreen'
}

# Load the data for the last 5 years
last_5_years = pd.read_csv('last_5_years.csv', delimiter=',')

# Load the data for each month of 2022
months_2022 = pd.read_csv('meses_2022.csv', delimiter=',')

# Set the time column as the index
last_5_years.set_index('Time', inplace=True)
months_2022.set_index('Time', inplace=True)

# Plot separate bar charts for each country
for country, color in country_colors.items():
    plt.figure(figsize=(10, 4))
    plt.bar(last_5_years.index, last_5_years[country], color=color)
    plt.title(f'Consumer Confidence Indicator Over the Last 5 Years - {country}')
    plt.xlabel('Year')
    plt.ylabel('Consumer Confidence')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for country, color in country_colors.items():
    plt.figure(figsize=(10, 4))
    plt.bar(months_2022.index, months_2022[country], color=color)
    plt.title(f'Consumer Confidence Indicator Over all months of 2022 - {country}')
    plt.xlabel('Year')
    plt.ylabel('Consumer Confidence')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


'''
########### ESTO ES INTERPRETANDO OTRA COSA DEL ENUNCIADO: TENGO PREGUNTAS

# Plot histograms for each country for the last 5 years
for country, color in country_colors.items():
    plt.hist(last_5_years[country], bins=20, edgecolor='k', alpha=0.6, color=color, label=country)
    
    # One histogram per country
    plt.title('Consumer Confidence Indicator for the Last 5 Years')
    plt.xlabel('Consumer Confidence')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    
    
# Only one histogram for all countries
plt.title('Consumer Confidence Indicator for the Last 5 Years')
plt.xlabel('Consumer Confidence')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()


# Plot histograms for each month of 2022
for country, color in country_colors.items():
    plt.hist(months_2022[country], bins=20, edgecolor='k', alpha=0.6, color=color, label=country)
    
    # One histogram per country
    plt.title('Consumer Confidence Indicator for every month of 2022')
    plt.xlabel('Consumer Confidence')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    
    
# Only one histogram for all countries
plt.title('Consumer Confidence Indicator for every month of 2022')
plt.xlabel('Consumer Confidence')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

'''