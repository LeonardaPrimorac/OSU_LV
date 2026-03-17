import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

data = data.dropna(axis=0)
data = data.drop_duplicates()    
data = data.reset_index(drop=True)

kat_stupci = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
data[kat_stupci] = data[kat_stupci].astype('category')

plt.figure()
plt.hist(data['CO2 Emissions (g/km)'], bins=30, color='cyan')
plt.title('Emisija CO2 plinova')
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Frekvencija')
plt.show()

plt.figure()
fuels = {'X':'blue', 'Z':'yellow', 'D':'green', 'E':'red', 'N':'purple'}
for fuel_type, color in fuels.items():
    subset = data[data['Fuel Type'] == fuel_type]
    plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'],
                color=color, label=fuel_type, s=10)
plt.legend(title='Fuel Type')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Odnos gradske potrošnje i emisije CO2')
plt.show()

data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.title('Razdioba izvangradske potrošnje po tipu goriva')
plt.suptitle('')
plt.xlabel('Fuel Type')
plt.ylabel('Fuel Consumption Hwy (L/100km)')
plt.show()

plt.figure()
cars_groupedbyfuel = data.groupby('Fuel Type').size()
cars_groupedbyfuel.plot(kind='bar', color='blue')
plt.title('Broj vozila po tipu goriva')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.show()

plt.figure()
cars_bycylinders = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
cars_bycylinders.plot(kind='bar', color='blue')
plt.title('Prosječna CO2 emisija po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna CO2 emisija (g/km)')
plt.show()