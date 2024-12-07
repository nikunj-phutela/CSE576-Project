import stanza
import math
import random
import llm_api
import pandas as pd

# Initialize Stanza NLP pipeline for English
nlp_mask = stanza.Pipeline('en')

# Function to randomly drop key terms to reduce context size
def drop_keyterms(key_terms):
    if len(key_terms) <= 4:
        return key_terms[:2]  # Keep at most 2 terms if list is short
    percentage_drop = 0.5  # Drop 50% of terms
    drop_count = math.ceil(len(key_terms) * percentage_drop)
    terms_to_drop = set(random.sample(key_terms, drop_count))
    # Keep remaining terms in their original order
    remaining_terms = [term for term in key_terms if term not in terms_to_drop]
    return remaining_terms

# Function to extract key terms from a question
def extract_terms(question):
    doc = nlp_mask(question)
    key_terms = [ent.text for ent in doc.ents]  # Extract named entities (NER)

    # Extract compound phrases and adjective-noun pairs using dependency parsing
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel in ['compound', 'amod']:  # Compound nouns and adjective modifiers
                compound_term = f"{word.text} {sentence.words[word.head - 1].text}"
                if compound_term not in key_terms:
                    key_terms.append(compound_term)

    # Add individual keywords (nouns, proper nouns, verbs, adjectives, adverbs)
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV']:
                if word.text not in key_terms:
                    key_terms.append(word.text)

    # Reduce the size of extracted terms
    terms_extract = drop_keyterms(key_terms)
    return terms_extract

# Function to generate relevant information using an LLM
def generate_relevant_info(llm_id: int, key_terms: list):
    terms = ", ".join(key_terms)  # Join terms into a string
    system_prompt = (
        "You are a helpful assistant that gives relevant information. "
        "Provide a comprehensive context or paragraph that meaningfully incorporates the given key terms or words."
    )
    user_prompt = f"""
      Provide relevant information and context incorporating the following key terms.
      Key terms: {terms}
      """
    # Use LLM API to generate the relevant information
    result = llm_api.use_llm(llm_id, system_prompt, user_prompt)
    return result

# Function to judge if generated information aligns with the expected answer
def judge_resp(answer, response):
    return llm_api.judge_response(answer, response)

# Main function to process a row of data and generate relevant info
def relevant_info(row, llm_id):
    # Extract key terms from the question
    extracted_terms = extract_terms(row['question'])
    row['extracted_terms'] = extracted_terms

    # Generate relevant information based on extracted terms
    generated_relevant_info = generate_relevant_info(llm_id, extracted_terms)
    row['rel_info_generated'] = generated_relevant_info

    # Judge whether the generated information matches the expected answer
    response = judge_resp(
        row['Best Answer'] if 'Best Answer' in row and pd.notna(row['Best Answer']) else row['answer'], 
        generated_relevant_info
    )

    return row, response
