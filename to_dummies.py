import pandas as pd

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("computers.csv")

for col in ['cd', 'laptop']:
        df[col].replace(['no', 'yes'], [0, 1], inplace=True)


# Guardar el resultado en un nuevo archivo CSV o sobrescribir el archivo original
df.to_csv("dummies.csv", index=False)
