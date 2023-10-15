import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

df = pd.read_csv("proteins_50000.csv")

#1) Reading the pattern from keyboard
chain =input("Introduce the pattern: ")

#2) Changes the pattern to UPPERCASE
upper = chain.upper()
print("The pattern you are looking for is:", upper)

#3) Start execution time
start = time.time()

#4) Reads the protein patterns from the file in pairs (id, sequence)
data = df.to_numpy()

#5) Look for occurrences of the pattern string inside each protein sequence
#6) If there are occurrences, register the id of the protein and the number of 
# occurrences in the sequence
results = []

for pair in data:
    index = pair[0]
    pattern = pair[1]

    if upper in pattern:
        num_times = pattern.count(upper)
        result = index, num_times
        results.append(result)
    
print(results)        

#7) Show the execution time
end = time.time()
exec_time = end - start
print("The execution time is:", exec_time)

#8) Print a histogram of occurrences using protein id as X and number of 
#occurrences as Y, using matplotlib.pyplot
#Represent the 10 proteins with more matches

#Sorting based on the number of matches
sorted_matches = sorted(results, key=lambda tupla: tupla[1], reverse=True)

#Keeping 10
more_matches = sorted_matches[:10]

# Desempaqueta los valores en X y Y
protein_id, num_occurences = zip(*more_matches)
protein_id = list(protein_id)
num_occurences = list(num_occurences)

#Pide esto pero creo que se confunde porque un histograma no devuelve el numero de occurences en y
#he intentado representar frecuencias absolutas así pero tampoco
#hist, bins = np.histogram(protein_id, bins=5)
#plt.hist(protein_id, bins=5, density = False)

plt.bar(protein_id, num_occurences) #sale vacío, he probado y con una lista de numeros pequeños va bien
#probad a ejecutarlo porque no me van spyder ni code e igual es eso

print(protein_id)
print(num_occurences)

plt.xlabel('Protein ID')
plt.ylabel('Num of occurences')
plt.show()


#9) Print the id and number of the protein with max occurrences
print('Max occurences protein', max(more_matches))