import random
import stanza
from fuzzywuzzy import fuzz
import llm_api

# Initialize Stanza NLP pipeline for English with specific processors
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

# Function to mask all words except nouns and randomly mask 50% of the nouns
def mask_non_nouns(doc):
    """
    Masks all non-noun words and randomly masks 50% of the nouns/adjectives/verbs.

    Args:
        doc: A processed document with .text and .xpos attributes for each word.
    
    Returns:
        tuple: (masked sentence, list of original unmasked nouns/adjectives/verbs)
    """
    masked_words = []
    noun_list = []
    noun_positions = []

    # Identify and mask words
    for sentence in doc.sentences:
        for i, word in enumerate(sentence.words): 
            if word.xpos.startswith('NN') or word.upos.startswith('ADJ') or word.upos.startswith('VERB'):
                masked_words.append(word.text)  # Keep the word
                noun_list.append(word.text)  # Add to the list of key terms
                noun_positions.append(i)  # Record the position
            else:
                masked_words.append('()')  # Mask non-nouns

    # Randomly select 50% of the nouns to mask
    num_nouns_to_mask = max(1, int(len(noun_list) * 0.5))
    if noun_positions:
        positions_to_mask = random.sample(noun_positions, num_nouns_to_mask)
        for pos in positions_to_mask:
            masked_words[pos] = '()'

    return ' '.join(masked_words), noun_list

# Function to check for similar questions using LLM completions
def check_similar_q(row, llm_id):
    """
    Checks if a similar question has been completed by the model.

    Args:
        row: Input row containing the question to evaluate.
        llm_id: Identifier for the LLM model.
    
    Returns:
        tuple: (row, bool indicating if a match was found)
    """
    similarity_threshold = 80  # Minimum similarity score for a match
    sentence = row['question']
    doc = nlp(sentence)

    # Mask the question and extract key terms
    mask_token = "()"
    masked_sentence, noun_list = mask_non_nouns(doc)
    mask_count = masked_sentence.count(mask_token)
    best_score = 0
    completions = []

    # Construct system and user prompts based on the LLM being used
    if llm_id in (2, 3):  # For Claude models
        system_prompt = (
            "You are a helpful assistant that completes masked words in questions. "
            f"Replace the {mask_count} masked word(s) marked with {mask_token} with suitable terms. "
            "Provide just the completed question."
        )
        user_prompt = f"Complete this question by replacing the masked sections:\n{masked_sentence}"
        for _ in range(5):
            response = llm_api.use_llm(llm_id, system_prompt, user_prompt, temp=1, top_k=5)
            completions.append(response)

    elif llm_id == 4:  # For Llama model
        system_prompt = (
            "Provide a natural and contextually appropriate question by completing the masked sections. "
            f"Replace the {mask_count} masked word(s) marked with {mask_token}. "
            "Provide just the completed question."
        )
        user_prompt = f"{system_prompt}\nComplete this question by replacing the masked sections:\n{masked_sentence}"
        completions = llm_api.use_llm(llm_id, system_prompt, user_prompt, temp=0.1, top_k=5)

    else:  # For GPT models
        system_prompt = "You are a helpful assistant that completes masked words in questions."
        user_prompt = f"""
        Complete the following question by replacing {mask_count} masked word(s) marked with {mask_token}.
        Question: {masked_sentence}
        """
        response = llm_api.use_llm(llm_id, system_prompt, user_prompt, temp=1, top_k=5)
        completions = [choice.message.content for choice in response.choices]

    # Compare each completion with the original question
    for completion in completions:
        score = fuzz.ratio(sentence, completion)
        if score > best_score:
            best_score = score

    matched = best_score >= similarity_threshold
    return row, matched
