import llm_api

# Function to obtain the predicted answer from the LLM
def obtain_answer(llm_id, row):
    # Prompt for the LLM to complete the answer based on the provided question-answer pair
    system_prompt = ""
    user_prompt = (
        f"Complete the incomplete option in the answer to the question in one word: "
        f"[{row['question_answer']}]. \nReply the answer only, the option without the full sentence."
    )

    # Use the LLM API to get the completion
    result = llm_api.use_llm(llm_id, system_prompt, user_prompt)

    # Store the predicted answer in the 'seen_terms' field of the row
    row['seen_terms'] = result
    return row

# Function to check if the generated answer matches the expected masked option
def match_answer(row):
    if row['masked_options'] and row['masked_options'][0].lower() == row['seen_terms'].lower():
        return True  # Match found
    return False  # No match

# Main function to check if the question and answer have been seen before
def seen_q_n_a(row, llm_id):
    # Obtain the answer and check if it matches the masked options
    result = obtain_answer(llm_id, row)
    return row, match_answer(result)
