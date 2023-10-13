import pandas as pd
df= pd.read_csv('computers.csv')
df2= df[8].eq('yes').mul(1)
print(df2)
