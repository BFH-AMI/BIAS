## For running simple experiments with different configurations

from utils import evaluate_combinations
import time
from datetime import datetime
import os

# Each combination has the form (test_name, metric, embedding, language, modelname)
combinations = [
   ('CW1_context', 'SEAT', 'bert', 'de', 'google-bert/bert-base-german-cased'),
   ('CW2_context', 'SEAT', 'bert', 'de', 'google-bert/bert-base-german-cased'),
   ('GER1_context', 'SEAT', 'bert', 'de', 'google-bert/bert-base-german-cased')
   ]

### Experiment Configurations
pval = False
iterations = 100
fullpermut = False

starttime = time.time()
logs = "pval=" + str(pval)+";iterations="+str(iterations)+";fullpermut="+str(fullpermut)+";starttime="+str(starttime)+"\n"
evaluated_results = evaluate_combinations(combinations, pval, iterations, fullpermut)
endtime = time.time()
duration = endtime-starttime
logs=logs + "endtime="+str(endtime)+"\nduration="+str(duration)+ "\n"
print("Time used for these experiments: "+str(duration))

resultsstring = ""
print("\n\nResults:\n" + '-'*50)
logs = logs + "\ncombination;pval;effectSize;pVal<0.05"+"\n"
for combination, result in evaluated_results.items():
    print(f"{combination}: {result}")
    if combination[1] != "CrowS_Pairs":
      if pval:
          print("p-value below threshold: " + str(result["p-value"]<0.05))
          resultsstring=resultsstring + (combination[0]+" & "+str(result["p-value"])+" & "+str(result["Effect Size"])+" & "+str(result["p-value"]<0.05)+"\\\ \hline"+"\n")
          logs = logs + str(combination) + ";" + str(result["p-value"])+";"+str(result["Effect Size"])+";"+str(result["p-value"]<0.05)+"\n"
print(resultsstring)
