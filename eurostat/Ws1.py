# Get the balance data for the Consumer confidence indicator,
#that is seasonally adjusted, for the EU: for the last
#5 years and for each month of 2022. Print histograms.

#VALUE OF ALL EUROPE AND VALUES FOR EACH COUNTRY
#LAST 5 YEARS, AND FOR EACH MONTH 2022
#TOTAL: 4 QUERIES

#https://ec.europa.eu/eurostat/databrowser/view/EI_BSCO_M__custom_7932751/default/table?lang=en

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
import matplotlib.pyplot as plt
import requests
import json
#from pyjstat import pyjstat

def plot_CCI(url, time_period):
    response = requests.get(url)
    data = response.text
    dic = json.loads(data)
    months = dic['dimension']['time']['category']['label']
    months_list = list(months.values())
    countries = dic['dimension']['geo']['category']['index']
    consumer_confidence = dic['value']
    
    num_months = len(months_list)
    num_countries = len(countries)
    
    for i in range(num_countries):
        plt.figure(figsize=(10, 4))
        country_name = list(countries.keys())[i]
        
        country_data = []
        for j in range(num_months):
            # Check if data is available for the combination of country and month
            country_index = i * num_months + j
            if str(country_index) not in consumer_confidence:
                # If data is missing, you can handle it here (e.g., skip, use a default value, etc.)
                # For this example, we'll add a None value to indicate missing data
                country_data.append(None)
            else:
                country_data.append(consumer_confidence[str(country_index)])
        #values belonging to RO are missing
        # Remove None values for plotting (optional)
        months_list_filtered = [month for month, data in zip(months_list, country_data) if data is not None]
        country_data_filtered = [data for data in country_data if data is not None]
        
        
        plt.bar(months_list_filtered, country_data_filtered, color='cyan')
        plt.title(f'Consumer Confidence Indicator {time_period}- {country_name}')
        plt.xlabel('Year - Month')
        plt.ylabel('Consumer Confidence')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
            
    

#REST request
# structure {host_url}/{service}/{version}/{response_type}/{datasetCode}?{format}&{lang}&{filters}
url_months_EU = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m?format=JSON&unit=BAL&indic=BS-CSMCI&s_adj=SA&lang=en&sinceTimePeriod=2022-01&untilTimePeriod=2022-12&geo=EU27_2020"
url_months_countries = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m?format=JSON&unit=BAL&indic=BS-CSMCI&s_adj=SA&lang=en&sinceTimePeriod=2022-01&untilTimePeriod=2022-12&geo=BE&geo=BG&geo=CZ&geo=DK&geo=DE&geo=EE&geo=IE&geo=EL&geo=ES&geo=FR&geo=HR&geo=IT&geo=CY&geo=LV&geo=LT&geo=LU&geo=HU&geo=MT&geo=NL&geo=AT&geo=PL&geo=PT&geo=RO&geo=SI&geo=SK&geo=FI&geo=SE&geo=UK&geo=ME&geo=MK&geo=AL&geo=RS&geo=TR"

url_last5y_EU = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m?format=JSON&unit=BAL&indic=BS-CSMCI&s_adj=SA&lang=en&sinceTimePeriod=2019-09&untilTimePeriod=2023-09&geo=EU27_2020"
url_last5y_countries = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m?format=JSON&unit=BAL&indic=BS-CSMCI&s_adj=SA&lang=en&sinceTimePeriod=2019-09&untilTimePeriod=2023-09&geo=BE&geo=BG&geo=CZ&geo=DK&geo=DE&geo=EE&geo=IE&geo=EL&geo=ES&geo=FR&geo=HR&geo=IT&geo=CY&geo=LV&geo=LT&geo=LU&geo=HU&geo=MT&geo=NL&geo=AT&geo=PL&geo=PT&geo=RO&geo=SI&geo=SK&geo=FI&geo=SE&geo=UK&geo=ME&geo=MK&geo=AL&geo=RS&geo=TR"

CCI_months_2022_EU = plot_CCI(url_months_EU, "in every month of 2022")
CCI_last5y_EU = plot_CCI(url_last5y_EU, "in the last 5 years")

CCI_months_2022_countries = plot_CCI(url_months_countries, "in every month of 2022")
CCI_last5y_countries = plot_CCI(url_last5y_countries, "in the last 5 years")
