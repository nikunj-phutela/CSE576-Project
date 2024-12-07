import llm_api

# Function to generate information for determining if a question has been seen before
def seen_q_info(row, llm_id, dataset_id):
    if dataset_id == 0:  # Special handling for dataset 0
        url = row['url']
        if isinstance(url, str):  # Remove the answer from the URL string
            url = url.replace(row['answer'], "")
            url = url.replace(row['answer'].lower(), "")
            url = url.replace(row['answer'].capitalize(), "")
        system_prompt = ""
        user_prompt = (
            f"Complete the sentence according to the hint in one word: [{row['masked_question']}]. "
            f"Hint:[{url}]. Reply the answer only in one word without full sentence."
        )
    else:  # General case for other datasets
        system_prompt = ""
        user_prompt = (
            f"Complete the sentence in one word by replacing the word in (): [{row['masked_question']}]. "
            f"Reply the answer only in one word without full sentence."
        )

    # Use the LLM to generate the answer
    result = llm_api.use_llm(llm_id, system_prompt, user_prompt)
    row['seen_terms'] = result  # Store the generated answer in the row
    return row

# Function to check if the generated answer matches the expected answer
def match_answer(row, llm_id, dataset_id):
    if llm_id in [0, 1, 4] and dataset_id == 0:  # Matching logic for specific models and dataset 0
        if row['answer'].lower() == row['seen_terms'].lower():
            return True
    elif llm_id in [2, 3] and dataset_id == 0:  # Special case for Claude models and dataset 0
        if (
            row['seen_terms'].replace(" ", "") == row['answer'] or
            row['seen_terms'].replace(" ", "").lower() == row['answer'].lower()
        ):
            return True
    else:  # General case for other datasets
        if row['masked_word_answer'].lower() == row['seen_terms'].lower():
            return True
    return False

# Main function to determine if a question has been seen before
def seen_q(row, llm_id, dataset_id):
    # Generate information and match it against the expected answer
    result = seen_q_info(row, llm_id, dataset_id)
    return row, match_answer(result, llm_id, dataset_id)
