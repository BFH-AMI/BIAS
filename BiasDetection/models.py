import os
import pickle
import numpy as np
import torch
from abc import ABC, abstractmethod
from transformers import BertTokenizer, AutoTokenizer

class EmbeddingModel(ABC):
    '''
    Abstract class for embedding models
    An instance of the class can be used to
     -load the model in RAM from storage (from formats: pickle, binary, txt, word2vec)
     -get the vector of a word
     -aggregate with the metric class (e.g. WEAT), to form an instance which allows to take a dataset and evaluate it
    '''

    LOADED_MODELS = {} # to keep track of loaded models, so we don't load them again. This is a class variable, so it is shared between all instances of the class. 
    LOADED_TOKENIZERS = {}

    def __init__(self, model_path, save_pickle=True, load_pickle=True): #model_path is the path to the model file, save_pickle and load_pickle to specify whether to save and load the model as a pickle file
        self.model_path = model_path #file with file ending e.g. "GoogleNews-vectors-negative300.bin" for word2vec
        self.model = None
        self.tokenizer = None
        self.save_pickle = save_pickle
        self.load_pickle = load_pickle
        self.language = 'en'
        self.device = None

    @abstractmethod #has to be implemented in subclasses. 
    def loading_model(self, language = 'en'):
        pass

    @abstractmethod
    def get_vector(self, word):
        pass

    def _save_as_pickle(self, filename):
        print(f"Saving model to {filename}")
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def _load_from_pickle(self, filename):
        print(f"Loading model from {filename}")
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)



class FasttextModel(EmbeddingModel):
    def loading_model(self, language):

        import fasttext # as imports are embedding-specific, we import them here

        if self.model_path in EmbeddingModel.LOADED_MODELS: # only load model if not already loaded
            print(f"Loading Fasttext model from cache")
            self.model = EmbeddingModel.LOADED_MODELS[self.model_path]
        else: 
            if self.load_pickle: 
                print("cannot use pickle for fasttext model")
            print(f"Loading Fasttext model from {self.model_path}")
            self.model = fasttext.load_model(self.model_path)
            EmbeddingModel.LOADED_MODELS[self.model_path] = self.model # save model in cache
        
    def get_vector(self, sequence):
        return np.mean([self.model.get_word_vector(word) for word in sequence.split()], axis=0)
        # return self.model.get_word_vector(word) # this is the original code, which only works for single words. The code above works for sequences of words.

class BertModel(EmbeddingModel):
    def __init__(self, model_path, save_pickle=False, load_pickle=False, save_model = True, embedding = '[CLS]'):
        '''
        model_path is the path to the model file, save_pickle and load_pickle aren't used for BERT models. 
        Instead use save_pretrained and from_pretrained methods.

        embedding = '[CLS]' , 'pooling' , 'first'  --  determines the embedding type for contextual word embeddings
        '''
        super().__init__(model_path, save_pickle, load_pickle)
        self.save_model = save_model
        self.embedding =  embedding

    def __call__(self, input_ids):
        return self.model.forward(input_ids)
      
    def loading_model(self, language = 'en', hidden_states = True):
        from transformers import BertModel, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel
        self.language = language

        if not hidden_states:
            self.model_path = self.model_path + '_logits'

        # Load model and tokenizer from cache
        if self.model_path in EmbeddingModel.LOADED_MODELS:
            self.model = EmbeddingModel.LOADED_MODELS[self.model_path]
            self.tokenizer = EmbeddingModel.LOADED_TOKENIZERS[self.model_path]
            print(f"Using bert model from cache at {self.model_path}") 
            

        else:
                local = os.path.exists(self.model_path)
                if local:
                    path = self.model_path
                else:
                    path = self.model_path.split('models/')[-1].replace('_logits', '')
                print(f"Loading bert model from {path}")
                if self.language == 'en':
                    self.model = BertForMaskedLM.from_pretrained(path, output_hidden_states = True)
                    self.tokenizer = BertTokenizer.from_pretrained(path.replace('models/', '', 1))

                elif self.language in ('no','is','tk','de','nl','it'):
                    if hidden_states:
                        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, output_hidden_states = True)
                        self.tokenizer = AutoTokenizer.from_pretrained(path.replace('models/', '', 1))
                    else:
                        self.model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True, output_hidden_states = True)
                        self.tokenizer = AutoTokenizer.from_pretrained(path.replace('models/', '', 1))
                elif self.language == 'multi':
                    self.model = BertModel.from_pretrained(path)
                    self.tokenizer = BertTokenizer.from_pretrained(path.replace('models/', '', 1))
                        
                EmbeddingModel.LOADED_MODELS[self.model_path] = self.model
                EmbeddingModel.LOADED_TOKENIZERS[self.model_path] = self.tokenizer

                # Save local copy of model
                if not local and self.save_model:
                    if hidden_states:
                        self.model.save_pretrained(self.model_path)
                    else:
                        self.model.save_pretrained(self.model_path + '_logits')

        ### E.g., Mac M2
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
        ### CUDA GPUs
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        except AttributeError:
            print("An error occured with torch device definition.")

        self.model.to(device)
        print("Model device:", next(self.model.parameters()).device)
        self.device = device
        self.model.eval()

    def get_vector(self, text, target_word = '[CLS]'):
        '''
        By default, returns the last hidden state correspond to the [CLS] token (as in May et al.).
        If you want to get the hidden state of a different word, specify it with the word argument.
        '''
        ## Roberta based tokenizers do not use CLS token. Instead, use <s>
        if 'Roberta' in self.tokenizer.__class__.__name__:
            target_word = '<s>'
        #print(target_word, text)
        #print(self.tokenizer.__class__)
        input_ids = torch.tensor(self.tokenizer.encode(text),device=self.device).unsqueeze(0)  # Batch size 1
        try:
            outputs = self.model(input_ids)
        except RuntimeError as e:
            print("Error during model forward pass:", e)

        embedding = self.embedding

        # Convert the word to (sub)tokens
        token = self.tokenizer.tokenize(target_word)

        if embedding == 'first':
            # If the token is broken into multiple subtokens, we take the first subtoken
            token = [token[0]]

        #print(target_word, token, text)
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        embeddings = []
        for token_id in token_ids:
            # Find index of the token in the input_ids
            token_index = input_ids[0].tolist().index(token_id)
            # Some models on HuggingFace don't return 'hidden_states' in the output.
            # In this case, we use 'last_hidden_state' instead.
            if 'hidden_states' in outputs.keys():
                last_hidden_states = outputs['hidden_states'][-1]
            else: 
                last_hidden_states = outputs['last_hidden_state']
            embedding = last_hidden_states[0][token_index].detach().cpu().numpy() # .cpu because of mps device type tensor cannot be converted
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        word_vector = np.mean(np.array(embeddings), axis = 0)
        return word_vector
