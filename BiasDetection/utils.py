from models import FasttextModel, BertModel
from metrics import WEAT, LPBS, CrowS_Pairs
import json
import pandas as pd

import config
# Folder to the models
folderpath = config.FOLDER_PATH
fasttextfile = config.FASTTEXT_FILE
word2vecfile = config.WORD2VEC_FILE
glovefile = config.GLOVE_FILE
mbertfile = config.mBERT_FILE

###
### This method evaluates each experiment passed in a "combinations" structure.
###
def evaluate_combinations(combinations, calc_pvalue_bool, p_value_iterations, full_permut_bool):
    results = {}
    for dataset, metric, embedding, language, modelname in combinations:
        print(f"\nEvaluating {dataset} with {metric} and {embedding} for language {language} with model {modelname}...\n")
        
        # Initialize the model
        if embedding == "word2vec":
            model = Word2VecModel(folderpath+word2vecfile)
        elif embedding == "glove":
            model = GloveModel(folderpath+glovefile)
        elif embedding == "fasttext":
            model = FasttextModel(folderpath+fasttextfile.replace("XX", language))
        # Bert models
        elif embedding == "bert":
            model = BertModel(folderpath+modelname)
        elif embedding == "bert_pooling":
            model = BertModel(folderpath+modelname, embedding = 'pooling')
        elif embedding == "bert_first":
            model = BertModel(folderpath+modelname, embedding = 'first')
        elif embedding == "mBERT":
            model = BertModel(folderpath+mbertfile)

        # API models
        elif embedding == "OpenAI_large":
            model = OpenAiModel(model_name = "text-embedding-3-large")
        elif embedding == "OpenAI_small":
            model = OpenAiModel(model_name = "text-embedding-3-small")
        elif embedding == "mVoyage":
            model = VoyageModel(model_name = "voyage-multilingual-2")


        # Load the model
        # TODO: Document why this exception only for Norwegian+LPBS
        if language == "no":
            if metric == "LPBS":
                model.loading_model(language="no", hidden_states = False)
            else:
                model.loading_model(language="no")
        else:
            model.loading_model(language=language)

        ### Case of using WEAT ####
        if metric == "WEAT":
            # Initialize the WEAT tester
            if embedding == "bert_pooling" or embedding == "bert_first":
                weat_tester = WEAT(model, enc = 'token-level')
            else:
                weat_tester = WEAT(model)
            
            # Load the selected dataset from file
            testfile = open('datasets/'+language+'/WEAT/'+dataset+'.txt', 'r')
            lines = testfile.read().split('\n')
            filtered_lines = [line for line in lines if not line.startswith('#')]
            dataset_obj = {}
            dataset_obj["Target1"] = filtered_lines[0].split(",")
            dataset_obj["Target2"] = filtered_lines[1].split(",")
            dataset_obj["Attribute1"] = filtered_lines[2].split(",")
            dataset_obj["Attribute2"] = filtered_lines[3].split(",")
            #print("Dataset_Obj:",dataset_obj)
            # Evaluate and store the result
            result = weat_tester.evaluate(dataset_obj, calc_pvalue_bool, p_value_iterations, full_permut_bool)
            results[(dataset, metric, embedding, language, modelname)] = result

        ### Case of using SEAT ###
        elif metric == "SEAT":
            # Initialize the SEAT tester
            if embedding == "bert_pooling" or embedding == "bert_first":
                seat_tester = WEAT(model, enc = 'token-level')
            else:
                seat_tester = WEAT(model)

            ## special case genSEAT
            if "genSEAT" in dataset:
                f = open('datasets/' + language + '/genSEAT/' + dataset + '.jsonl', 'r')
            else:
                f = open('datasets/'+language+'/SEAT/'+dataset+'.jsonl', 'r')
            testfile = json.load(f)

            dataset_obj = {}
            dataset_obj["Target1"] = testfile["targ1"]["examples"]
            dataset_obj["Target2"] = testfile["targ2"]["examples"]
            dataset_obj["Attribute1"] = testfile["attr1"]["examples"]
            dataset_obj["Attribute2"] = testfile["attr2"]["examples"]
            #print("Dataset_Obj:",dataset_obj)

            # Evaluate and store the result
            result = seat_tester.evaluate(dataset_obj, calc_pvalue_bool, p_value_iterations, full_permut_bool)
            results[(dataset, metric, embedding, language)] = result

        ### Case of using LPBS ###
        elif metric == "LPBS":
            # Initialize the LPBS tester
            LPBS_tester = LPBS(model)

            f = open('datasets/'+language+'/LPBS/'+dataset+'.jsonl', 'r')
            testfile = json.load(f)

            dataset_obj = {}
            dataset_obj["Target1"] = testfile["targ1"]
            dataset_obj["Target2"] = testfile["targ2"]
            dataset_obj["Attribute1"] = testfile["attr1"]
            dataset_obj["Attribute2"] = testfile["attr2"]
            dataset_obj["Templates"] = testfile["templates"]
            #print("Dataset_Obj:",dataset_obj)

            # Evaluate and store the result
            result = LPBS_tester.evaluate(dataset_obj, calc_pvalue_bool, p_value_iterations, full_permut_bool)
            results[(dataset, metric, embedding, language, modelname)] = result

        ### Case of using CrowS Metric ###
        elif metric == "CROWS":
            # Initialize the CrowS_Pairs tester
            crows_pairs_tester = CrowS_Pairs(model)

            df = pd.read_csv('datasets/'+language+'/CROWS/'+dataset+'.csv')

            result = crows_pairs_tester.evaluate(df)
            results[(dataset, metric, embedding, language, modelname)] = result

    return results