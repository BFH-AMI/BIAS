import numpy as np
import random
from metric_helper_functions import effect_size_embed, pred_prob_for_mask, pll_scores
from transformers import BertTokenizer, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
#from identifyTokenSEAT import match_templates

###
### WEAT and WEAT-based metrics
###

class WEAT:
    def __init__(self, model, enc = 'default'): # model is needed as it contains the method to get the vector of a word: get_vector(word)
        '''
        enc: encoding type, can be 'word' or 'sent'
        This is only important for BERT models, where the encoding can be associated to either
        the [CLS] ('default') token or targ./attr. ('token-level') token.
        '''
        self.model = model
        self.enc = enc
        self.language = model.language

     
    def get_word_embeddings(self, words, seed_words = None):
        '''
        For a given list of words, returns the embeddings of the words.
        '''
        enc = self.enc
        if enc == 'default':
            return [self.model.get_vector(word) for word in words] # get_vector from model as it is diferent for each model
        elif enc == 'token-level':
            return [self.model.get_vector(text = sentence, target_word = seed_word) for (sentence, seed_word) in zip(words, seed_words)]

    def get_target_tokens(self, texts):
        '''
        For a given word list or sentence list, extracts the target tokens for computing
        token-level contextual word embeddings.
        '''
        # Test if texts are sentences (SEAT) or words (WEAT)
        if "." in texts[0]:
            matches = match_templates(texts, language = self.language)
            seed_words = [match[1] for match in matches.values()]
        else:
            seed_words = texts
        return seed_words

    def get_target_attribute_embeddings(self, target1, target2, attribute1, attribute2):
        enc = self.enc
        # For token-level embeddings, need to specify the token of interest
        # for each sentence in the dataset.
        lists = {'Target1': target1, 'Target2': target2, 'Attribute1': attribute1, 'Attribute2': attribute2}
        embeddings = {}
        for key, list in lists.items():
            seed_words = None
            if enc == 'token-level':
                # Extracts target token from template.
                seed_words = self.get_target_tokens(list)
            embeddings[key] = self.get_word_embeddings(list, seed_words)

        target1_embed = embeddings['Target1']
        target2_embed = embeddings['Target2']
        attribute1_embed = embeddings['Attribute1']
        attribute2_embed = embeddings['Attribute2']
        
        return target1_embed, target2_embed, attribute1_embed, attribute2_embed
    

    def effect_size(self, embeddings):
        target1_embed, target2_embed, attribute1_embed, attribute2_embed = embeddings
        return effect_size_embed(target1_embed, target2_embed, attribute1_embed, attribute2_embed)
    
    def permutation_test(self, embeddings, num_permutations):
        target1_embed, target2_embed, attribute1_embed, attribute2_embed = embeddings
        combined = target1_embed + target2_embed
        size = len(target1_embed) # both lists have the same size
        count = 1
        observed_test_stat = effect_size_embed(target1_embed, target2_embed, attribute1_embed, attribute2_embed)

        print('calculating p-value, % of permutations done: ')
        for i in range(num_permutations):
            random.shuffle(combined)
            new_target1 = combined[:size]
            new_target2 = combined[size:]

            test_stat = effect_size_embed(new_target1, new_target2, attribute1_embed, attribute2_embed)
            if test_stat > observed_test_stat:
                count += 1


            # display progress
            if (i+1) % ((num_permutations) / 10) == 0:
                print(f'{(i+1) / (num_permutations / 100)}%, ', end='')
        print('\n')

        return count / num_permutations

    def evaluate(self, dataset, calc_pval, number_permut, full_permut): # included the option to not calculate the p-value, as it takes a long time
        target1 = dataset["Target1"]
        target2 = dataset["Target2"]
        attribute1 = dataset["Attribute1"]
        attribute2 = dataset["Attribute2"]
    

        embeddings = self.get_target_attribute_embeddings(target1, target2, attribute1, attribute2)
        
        effect = self.effect_size(embeddings)
        print(f'effect size: {effect:.2f}')
        p_value = None
        if calc_pval:
            if full_permut:
                print('Full permutation test is currently not implemented.')
            else:
                p_value = self.permutation_test(embeddings, num_permutations=number_permut)
            
            print("p-value: ", p_value)
        
        return {
            "Effect Size": effect,
            "p-value": p_value
        }

###
### LPBS Metric (Kurita et al.)
###

class LPBS:
    '''
    Original implementation: https://github.com/keitakurita/contextual_embedding_bias_measure/blob/master/notebooks/weat_result_replication_permute_targets.ipynb
    '''
    def __init__(self, model, ):
        self.model = model
        self.language = model.language
        if self.language == 'en':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.language == 'no':
            self.tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-xs")

    
    def p_tgt(self, template, target, attribute):
        '''
        Replace target in sentence of the form "TARGET is ATTRIBUTE" with [MASK] and calculate the probability that the [MASK] token is the target word.
        '''
        sentence = template.replace("ATTRIBUTE", attribute)
        sentence_masked = sentence.replace("TARGET", "[MASK]")
        return pred_prob_for_mask(self.model, self.tokenizer, sentence_masked, target)
    
    def p_prior(self, template, target):
        '''
        Replace target and attribute in sentence of the form "TARGET is ATTRIBUTE" with [MASK],
        and calculate the probability that the [MASK] token corresponding to TARGET is the target word.

        NOTE: Although p_tgt computes the same values as in the Kurita implementation, the p_prior values are different.
        '''
        sentence = template.replace("TARGET", target).replace("ATTRIBUTE", "[MASK]")
        sentence_masked = sentence.replace(target, "[MASK]")
        return pred_prob_for_mask(self.model, self.tokenizer, sentence_masked, target)

    def inc_log_prob(self, template, target, attribute):
        '''
        Calculate the increased log probability score for a sentence of the form "TARGET is ATTRIBUTE".
        '''
        p_tgt = self.p_tgt(template, target, attribute)
        p_prior = self.p_prior(template, target)
        return np.log(p_tgt / p_prior)

    def log_bias_score(self, w, t_1, t_2, template):
        '''
        Calculate the log bias score of a word w for a pair of target words and a template.
        '''
        return self.inc_log_prob(template, t_1, w) - self.inc_log_prob(template, t_2, w)

    def bias_score_attribute(self, attribute, target1, target2, templates):
        '''
        Calculate the bias score of an attribute word, averaging over all pairs of target words and templates.
        '''
        # # For plural templates
        # log_bias_scores_pl = [[self.log_bias_score(attribute, t_1, t_2, template) for template in templates['plural']] for (t_1, t_2) in zip(target1['plural'], target2['plural'])]
        # # convert to numpy array
        # log_bias_scores_pl = np.array(log_bias_scores_pl)
        # log_bias_score_pl = np.mean(log_bias_scores_pl)

        # For singular templates
        log_bias_scores = [[self.log_bias_score(attribute, t_1, t_2, template) for template in templates['singular']] for (t_1, t_2) in zip(target1['singular'], target2['singular'])]
        # convert to numpy array
        log_bias_scores = np.array(log_bias_scores)
        log_bias_score = np.mean(log_bias_scores)

        # bias_score = (log_bias_score_pl + log_bias_score_sg) / 2
        bias_score = log_bias_score

        return bias_score
    
    def effect_size_log(self, target1, target2, attribute1, attribute2, templates):
        '''
        Calculate the effect size for a set of target words and attribute words.
        '''
        # Compute the bias score for each attribute word
        log_bias_scores_a1 = [self.bias_score_attribute(a, target1, target2, templates) for a in attribute1]
        log_bias_scores_a2 = [self.bias_score_attribute(a, target1, target2, templates) for a in attribute2]

        df_1 = pd.DataFrame({'log_bias_scores': log_bias_scores_a1, 'attribute': attribute1})
        df_2 = pd.DataFrame({'log_bias_scores': log_bias_scores_a2, 'attribute': attribute2})
        print(df_1)
        print(df_2)

        mean_attribute1 = np.mean(log_bias_scores_a1)
        mean_attribute2 = np.mean(log_bias_scores_a2)

        # Compute the standard deviation over all attribute words.
        # This differs from WEAT, where the standard deviation is computed over all target words.
        # The justification for this in Kurita et al (2019) is that they use only a small subset of the target words.
        stdev = np.std(log_bias_scores_a1 + log_bias_scores_a2)

        return (mean_attribute1 - mean_attribute2) / stdev
    
    def evaluate(self, dataset, calc_pval, number_permut, full_permut = False): # included the option to not calculate the p-value, as it takes a long time
        target1 = dataset['Target1']
        target2 = dataset['Target2']
        attribute1 = dataset['Attribute1']
        attribute2 = dataset['Attribute2']
        templates = dataset['Templates']
        
        effect = self.effect_size_log(target1, target2, attribute1, attribute2, templates)
        print(f'effect size: {effect:.2f}')
        p_value = None
        if calc_pval:
            if full_permut:
                print('Too many permutations, defaulting to 10000')
                p_value = self.permutation_test(target1, target2, attribute1, attribute2, templates, num_permutations=10000)
            else:
                p_value = self.permutation_test(target1, target2, attribute1, attribute2, templates, num_permutations=number_permut)
            
            print("p-value: ", p_value)
        
        return {
            "Effect Size": effect,
            "p-value": p_value
        }
    
    def permutation_test(self, target1, target2, attribute1, attribute2, templates, num_permutations):
        '''
        Computes the one-sided p-value for log-WEAT. This differs from the original implementation, which computes a two-sided p-value.
        This also differs from WEAT, since partitions of atribute words are used instead of target words.
        '''
        observed_test_stat = self.effect_size_log(target1, target2, attribute1, attribute2, templates)
        
        # Compute the bias score for each attribute word
        # This only needs to be done once, as the bias scores are the same for each permutation.
        log_bias_scores_a1 = [self.bias_score_attribute(a, target1, target2, templates) for a in attribute1]
        log_bias_scores_a2 = [self.bias_score_attribute(a, target1, target2, templates) for a in attribute2]
        stdev = np.std(log_bias_scores_a1 + log_bias_scores_a2)

        df_1 = pd.DataFrame({'log_bias_scores': log_bias_scores_a1, 'attribute': attribute1})
        df_2 = pd.DataFrame({'log_bias_scores': log_bias_scores_a2, 'attribute': attribute2})

        print('calculating p-value, % of permutations done: ')
        count = 1
        for i in range(num_permutations):
            # Concatenate df1 and df2
            df = pd.concat([df_1, df_2])
            # Shuffle the rows
            df = df.sample(frac=1).reset_index(drop=True)
            # Split the dataframe
            df_1 = df.iloc[:len(attribute1)]
            df_2 = df.iloc[len(attribute1):]
            
            mean_1 = np.mean(df_1['log_bias_scores'])
            mean_2 = np.mean(df_2['log_bias_scores'])

            test_stat = (mean_1 - mean_2) / stdev

            if test_stat > observed_test_stat:
                count += 1


            # display progress
            if (i+1) % ((num_permutations) / 10) == 0:
                print(f'{(i+1) / (num_permutations / 100)}%, ', end='')
        print('\n')

        return count / num_permutations

###
### CrowS Metric
###

class CrowS_Pairs:
    '''
    Adapted from original code: https://github.com/nyu-mll/crows-pairs/blob/master/metric.py
    '''
    def __init__(self, model, ):
        self.model = model
        self.language = model.language
        if self.language == 'en':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.language == 'no':
            self.tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-xs")
    
    def evaluate(self, dataset):
        tokenizer = self.tokenizer
        model = self.model
        if torch.cuda.is_available():
            model.to('cuda')

        df = dataset
        # df = dataset.iloc[4:5] # for testing purposes
        df_score = pd.DataFrame(columns=['sent_more', 'sent_less', 
                                    'sent_more_score', 'sent_less_score',
                                    'score', 'stereo_antistereo', 'bias_type'])
        N = 0
        neutral = 0
        total_stereo, total_antistereo = 0, 0
        stereo_score, antistereo_score = 0, 0
        
        print('Calculating CrowS-Pairs metric...')
        with tqdm(total=len(df)) as pbar:
            for i, row in df.iterrows():
                sent1 = row['sent_more']
                sent2 = row['sent_less']
                bias_direction = row['stereo_antistereo']
                bias = row['bias_type']

                pll_1, pll_2 = pll_scores(sent1, sent2, tokenizer, model)
                N += 1
                pair_score = 0
                pbar.update(1)

                if pll_1 == pll_2:
                    neutral += 1

                else:
                    if bias_direction == 'stereo':
                        total_stereo += 1
                        if pll_1 > pll_2:
                            stereo_score += 1
                            pair_score = 1
                    elif bias_direction == 'antistereo':
                        total_antistereo += 1
                        if pll_1 > pll_2:
                            antistereo_score += 1
                            pair_score = 1
                
                df_score = pd.concat([df_score, pd.DataFrame([{'sent_more': sent1,
                            'sent_less': sent2,
                            'sent_more_score': pll_1,
                            'sent_less_score': pll_2,
                            'score': pair_score,
                            'stereo_antistereo': bias_direction,
                            'bias_type': bias
                            }])], ignore_index=True)

        df_score.to_csv('crows_results.csv')
        metric_score = (stereo_score + antistereo_score) / N * 100
        stereo_metric = stereo_score / total_stereo * 100

        print('Total examples:', N)
        print('Metric score:', round(metric_score, 2))
        print('Stereotype score:', round(stereo_metric, 2))
        if antistereo_score != 0:
            antistereo_metric = antistereo_score / total_antistereo * 100
            print('Anti-stereotype score:', round(antistereo_metric, 2))
        print("Num. neutral:", neutral, round(neutral / N * 100, 2))
        
        return metric_score