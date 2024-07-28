from utils import evaluate_combinations
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt

### Select here your experiments configuration folder, see an example in folder experiments/example
#folder_path = 'experiments/example/'
folder_path = 'experiments/experiments-DE/'


### Give your experiment a name, this will be used for the folder where the results are stored
#experiment_name = "test1"
experiment_name = "GermanTest"

### Experiment Configurations
pval = True
iterations = 100
fullpermut = False

## Iterare through all experiments in the selected subfolder.
## Execute each experiment, create log file and plot.
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    combinations = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                combinations.append(parts)
    print(combinations)

    starttime = time.time()
    logs = "pval=" + str(pval)+";iterations="+str(iterations)+";fullpermut="+str(fullpermut)+";starttime="+str(starttime)+"\n"
    evaluated_results = evaluate_combinations(combinations, pval, iterations, fullpermut)
    endtime = time.time()
    duration = endtime-starttime
    logs=logs + "endtime="+str(endtime)+"\nduration="+str(duration)+ "\n"
    print("Time used for these experiments: "+str(duration))

    resultsstring = ""
    print("\n\nResults:\n" + '-'*50)
    logs = logs + str(combinations)
    logs = logs + "\ncombination;pval;effectSize;pVal<0.05"+"\n"
    for combination, result in evaluated_results.items():
        print(f"{combination}: {result}")
        if combination[1] != "CrowS_Pairs":
            if pval:
                print("p-value below threshold: " + str(result["p-value"]<0.05))
                resultsstring=resultsstring + (combination[0]+" & "+str(result["p-value"])+" & "+str(result["Effect Size"])+" & "+str(result["p-value"]<0.05)+"\\\\ \\hline"+"\n")
                logs = logs + str(combination) + ";" + str(result["p-value"])+";"+str(result["Effect Size"])+";"+str(result["p-value"]<0.05)+"\n"
    print(resultsstring)

    logs=logs + "\n \n \n Latex Tables \n \n"

    header = """\\begin{table}[h]
    \\centering
    \\small
    \\begin{tabular}{|p{4cm}|p{3cm}|p{2cm}|p{2cm}|}
        \\hline
        \\textbf{Name} & \\textbf{p-value} & \\textbf{Effect Size} & \\textbf{Bias \\mbox{Detected?}} \\\\
        \\hline \n"""

    ending = """\\end{tabular}
  \\caption{""" +str(experiment_name+":"+filename)+""".}
  \\label{table-results}
\\end{table}"""

    resultsstring = resultsstring.replace("True","\\cmark ")
    resultsstring = resultsstring.replace("False","\\xmark ")
    resultsstring = resultsstring.replace("_", "\\_")
    print(header)
    print(resultsstring)
    print(ending)

    logs = logs + header + resultsstring + ending
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    foldername = f"{experiment_name}"
    if not os.path.exists("results/"+foldername):
        os.makedirs("results/"+foldername)
    filename2 = filename.replace(".txt","_") + f"{current_date}_experiment.log"
    with open("results/" + foldername + "/" + filename2, "w") as file:
        file.write(logs)

    ## Generate plots
    # Separate the results into their components
    combinations, effect_sizes, p_values = zip(*[(k, v['Effect Size'], v['p-value']) for k, v in evaluated_results.items()])
    # Convert effect sizes to point sizes (you can adjust the factor for your needs)
    point_sizes = [1000 * abs(e) for e in effect_sizes]
    # Plot
    plt.figure(figsize=(15, 8))

    # Scatter plot with combinations on the x-axis and p-values on the y-axis
    plt.scatter(range(len(combinations)), p_values, s=point_sizes, alpha=0.6)

    # Drawing a horizontal line for the significance threshold
    plt.axhline(y=0.05, color='r', linestyle='-')
    plt.axhline(y=0.00, color='k', linestyle='--')

    # Setting the x-ticks to be the combination names and rotating them for better visibility
    plt.xticks(range(len(combinations)), combinations, rotation=90)

    upper_limit = max(0.1, (max(p_values) + 0.1))  # showing at least until 0.1, even if all p-values are smaller.
    plt.ylim([-0.02, upper_limit])

    plt.xlabel("Combinations")
    plt.ylabel("P-Value")
    plt.title("P-Value vs Combinations with Effect Size as Point Size")

    # Displaying the plot
    plt.tight_layout()

    plt.savefig("results/"+foldername+f"/"+filename.replace(".txt","")+"_"+str(current_date)+"_plot.png")
