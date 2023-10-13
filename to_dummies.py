import pandas as pd

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("computers.csv")

# Seleccionar la quinta columna (índice 4) y dividir los valores separados por comas
print(df.iloc[:,7])
print(df.iloc[:,8])

df.iloc[:, 7] = df.iloc[:, 7].replace({'yes': 1, 'no': 0})
df.iloc[:, 8] = df.iloc[:, 8].replace({'yes': 1, 'no': 0})


# Guardar el resultado en un nuevo archivo CSV o sobrescribir el archivo original
df.to_csv("dummies.csv", index=False)
