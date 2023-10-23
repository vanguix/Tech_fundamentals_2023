# Get Statistical information

'''
- Service: onomastica
- Main version: 1
- Operation: nadons
- Format: json
- Parameters:
- id=40683
- lang=en
'''

import requests
import json
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

#########PART 1: Women born in Catalonia in the last 5 years

url = 'https://www.idescat.cat/pub/?id=naix&n=364&lang=es'
headers= "Mozilla/5.0"

##########There is no service for this so I use the website with the data
def scrape_data(url, headers):
    nens, nenes, total = [], [], []

    web_browsers = {'User-Agent': headers}
    response = requests.get(url, headers=web_browsers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, features='html.parser')

        # Find the table
        table = soup.find(name='table', class_='ApartNum xs Cols4')

        # Extract data
        rows = table.tbody.find_all('tr')
        data = []
        for row in rows:
            columns = row.find_all('td')
            row_data = [column.text.strip() for column in columns]
            data.append(row_data)
        nens = sorted([int(x[0].replace('.', '')) for x in data][0:5], reverse=True)
        nenes = sorted([int(x[1].replace('.', '')) for x in data][0:5], reverse=True)
        total = sorted([int(x[2].replace('.', '')) for x in data][0:5], reverse=True)
    else:
        print(f'Failed to retrieve the website. Status code: {response.status_code}')
    return nens, nenes, total

nens, nenes, total = scrape_data(url, headers)

last_5y = [str(year) for year in range(2018, 2023)]

#Number of women born in Calalonia in every of the last 5 years
for i in range(len(last_5y)):
    print('Year:', last_5y[i], ', Women born:', nenes[i])

#####Total number of women born in Catalonia in the last 5 years
total_nenes = sum(nenes)
print('Total number of women born in Catalonia in the last 5 years:', total_nenes)

plt.figure(figsize=(10, 6))

plt.bar(last_5y, nenes, color='skyblue')
plt.xlabel('Last 5 years')
plt.ylabel('Number of women born')
plt.title('Women born in Catalonia in the last 5 years')
plt.xticks(rotation=45) 

plt.tight_layout()  # Ajustar el diseño para que quepa todo

plt.show()

#########PART 2: Women named María born in Catalonia in the last 5 years

# url = http://api.idescat.cat/{service}/v{main version}/{operation}.{format}[?param&param…]
url2 = 'https://api.idescat.cat/onomastica/v1/nadons/dades.json?id=40683&class=t&lang=en'
response2 = requests.get(url2)
data2= response2.text #JSON dictionary

dic2 = json.loads(data2)
#20 corresponds to year 2017

years= [] #list of years for plotting later
marias_years = []
indexes_years = range(21,26) #indexes with data from the last 5 years

for i in indexes_years:
    year = dic2["onomastica_nadons"]["ff"]["f"][i]["c"]
    years.append(year)
    
for i in indexes_years:
    marias_number = dic2["onomastica_nadons"]["ff"]["f"][i]["pos1"]["v"]
    marias_years.append(int(marias_number))

#Number of women born named Maria in every of the last 5 years
for i in range(len(years)):
    print('Year:', years[i], ', Women born named María:', marias_years[i])

#Total number of women born named Maria in the last 5 years
total = sum(marias_years)
print('Women born in Catalonia named Maria in the last 5 years: ', total)

plt.bar(years, marias_years, color='skyblue')
plt.xlabel('Last 5 years')
plt.ylabel('Women born named María')
plt.title('Women born named María in the last 5 years (2018-2022)')
plt.show()

########PART 3: Mujeres nacidas en los últimos 5 años que se llaman María por ciudad

def fetch_data(years):
    dic_list = []
    base_url = 'https://api.idescat.cat/onomastica/v1/nadons/dades.json?id=40683&t={}&class=com&lang=en'
    
    for year in years:
        url = base_url.format(year)
        response = requests.get(url)
        data = response.text
        dic = json.loads(data)
        dic_list.append(dic)
    
    return dic_list

def analyze_data(dic_list, years):
    location_indexes = range(0, 42)
    most_populous_counties = []

    for x, j in enumerate(dic_list):  # For every year
        counties = []
        marias_locations = []
        marias_per_county = []

        

        most_populous_county = ""
        number_of_marias = 0

        for i in location_indexes:  # For every location
            county = j["onomastica_nadons"]["ff"]["f"][i]["c"]["content"]
            counties.append(county)

            if j["onomastica_nadons"]["ff"]["f"][i]["pos1"]["v"] != "_":
                marias_number = j["onomastica_nadons"]["ff"]["f"][i]["pos1"]["v"]
                marias_locations.append(int(marias_number))
                if int(marias_number) > number_of_marias:  # Check if it's the most populous
                    most_populous_county = county
                    number_of_marias = int(marias_number)
            else:
                marias_locations.append(0)
                
            marias_per_county.append(marias_locations) #info of every county per year
            

        #print(f'--------------------------------------------Year: {years[x]}')

        most_populous_counties.append((years[x], most_populous_county, number_of_marias))

        # Number of women born named Maria in every county of the last 5 years
        #for i in range(len(counties)):
            #print('County:', counties[i], ', Women born named María:', marias_locations[i])

    total_marias_loc = [sum(k) for k in zip(*marias_per_county)]

    plot_data(counties, total_marias_loc)

    return most_populous_counties

def plot_data(counties, marias_locations):
    plt.bar(counties, marias_locations,  color='skyblue')
    plt.xlabel('Counties')
    plt.ylabel('Women born named María')
    plt.title('Women born named María in every county in the last 5 years')
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.tight_layout()  # Ajustar el diseño para que quepa todo
    plt.show()

def display_results(most_populous_counties):
    print("\nCounties with more Marias born per year:")
    print("Year  |  County  |  Number of Marias born")
    print("--------------------------------------------")
    for year, county, number_of_marias in most_populous_counties:
        print(f"{year}  |  {county}  | {number_of_marias}")
        
def bar_plot(most_populous_countries):
    plt.figure(figsize=(10, 6))
    years_counties, numbers_of_marias = zip(*[(f"{year} - {county}", number_of_marias) for year, county, number_of_marias in most_populous_counties])

    plt.bar(years_counties, numbers_of_marias, color='skyblue')
    plt.xlabel('County with the max number of Marias born in a year')
    plt.ylabel('Max number of Marias born')
    plt.title('County with the max number of Marias born per year')
    plt.xticks(rotation=45) 

    plt.tight_layout()  # Ajustar el diseño para que quepa todo

    plt.show()

dic_list = fetch_data(years)

most_populous_counties = analyze_data(dic_list, years)
display_results(most_populous_counties)
bar_plot(most_populous_counties)

