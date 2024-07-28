import numpy as np
import torch
import difflib

###
### Helper functions for WEAT
###

def cosine_similarity(v1, v2):
    """
    Cosine similarity between two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def s(w_embed, A_embed, B_embed):
    """
    Measurement of association between one target word and two attribute words
    """
    return np.mean([cosine_similarity(w_embed, a) for a in A_embed]) - np.mean([cosine_similarity(w_embed, b) for b in B_embed])

def effect_size_embed(target1_embed, target2_embed, attribute1_embed, attribute2_embed):
    """
    Modified version of effect_size, which takes the embeddings as input instead of the words. 
    This is used to speed up the permutation test, as the embeddings only have to be calculated once.
    """
    mean_target1 = np.mean([s(t, attribute1_embed, attribute2_embed) for t in target1_embed]) # asspciation of all target1 with attribute1 and attribute2
    mean_target2 = np.mean([s(t, attribute1_embed, attribute2_embed) for t in target2_embed])
    stdev = np.std([s(t, attribute1_embed, attribute2_embed) for t in target1_embed + target2_embed]) #std over all: target1 and target2
    return (mean_target1 - mean_target2) / stdev

###
### Helper functions for Kurita et al. (2019).
###

def pred_prob_for_mask(model, tokenizer, masked_sentence, target, ids = False):
    '''
    Compute probability of [MASK] token being target. Note: This assumes the mask token corresponding to the target word is the first [MASK] token in the sentence.
    '''
    if ids == False:
        input_ids = torch.tensor(tokenizer.encode(masked_sentence), device=model.device).unsqueeze(0)
    else:
        input_ids = masked_sentence.unsqueeze(0)
    # Get the index of the first [MASK] token in the input_ids tensor.
    mask_index = input_ids[0].tolist().index(tokenizer.mask_token_id)

    # Get logits for the [MASK] token corresponding to target.
    output = model(input_ids) # This is a tuple consisting of (logits, hidden_states)
    logits = output[0] # logits is a tensor of shape (batch_size, sequence_length, vocab_size), so this corresponds to the only sentence in the batch.
    target_token_ids = tokenizer(target)['input_ids']
    # print(target_token_ids[1:-1])
    
    logits_for_mask = logits[0][mask_index]

    # Get predicted probabilities for the [MASK] token.
    predicted_probabilities = torch.nn.functional.softmax(logits_for_mask, dim=-1)

    # Get the predicted probability for the target word. For subtokens, multiply the probabilities.
    predicted_prob_for_target = 1
    for target_token_id in target_token_ids[1:-1]:
        predicted_prob_for_target *= predicted_probabilities[target_token_id]

    return predicted_prob_for_target.item()

###
### Helper functions for CrowS-Pairs.
###

def get_shared_token_indices(sent1_ids, sent2_ids):
    """
    Returns the indices of the shared tokens between two sentences. 
    To account for possible wordpiece issues, return indices for each sentence separately.
    """

    seq1 = [str(x) for x in sent1_ids.tolist()]
    seq2 = [str(x) for x in sent2_ids.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2

def pll_scores(sent1, sent2, tokenizer, model):
    """
    This function computes the pseudo-log-likelihood scores for each of the sentences
    in a given pair.

    First compute predicted probabilities for each sentence by masking one shared word at a time.

    The score each sentence by computing the sum of the log of the predicted probabilities.
    """
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sent1 = sent1.lower()
    sent2 = sent2.lower()

    # Tokenize the sentences.
    sent1_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_ids = tokenizer.encode(sent2, return_tensors='pt')

    # Get the indices of non-changing tokens between the two sentences.
    shared_ind1, shared_ind2 = get_shared_token_indices(sent1_ids[0], sent2_ids[0])

    assert len(shared_ind1) == len(shared_ind2)
    num_shared_tokens = len(shared_ind1)

    # Compute mask predictions as mask token is moved through the shared tokens.
    mask_id = tokenizer.mask_token_id
    sent_1_probs = []
    sent_2_probs = []

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1,num_shared_tokens-1):
        ground_truth1 = tokenizer.convert_ids_to_tokens([sent1_ids[0][shared_ind1[i]]])
        ground_truth2 = tokenizer.convert_ids_to_tokens([sent2_ids[0][shared_ind2[i]]])

        masked_sent1_ids = sent1_ids[0].clone().detach()
        masked_sent2_ids = sent2_ids[0].clone().detach()

        masked_sent1_ids[shared_ind1[i]] = mask_id
        masked_sent2_ids[shared_ind2[i]] = mask_id
        
        score1 = np.log(pred_prob_for_mask(model, tokenizer, masked_sent1_ids, ground_truth1, ids = True))
        score2 = np.log(pred_prob_for_mask(model, tokenizer, masked_sent2_ids, ground_truth2, ids = True))

        sent_1_probs.append(score1)
        sent_2_probs.append(score2)

    # Compute the pseudo-log-likelihood scores.
    pll_score1 = torch.sum(torch.tensor(sent_1_probs)).item()
    pll_score2 = torch.sum(torch.tensor(sent_2_probs)).item()

    return pll_score1, pll_score2