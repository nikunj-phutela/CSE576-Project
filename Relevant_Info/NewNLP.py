from openai import OpenAI
import pandas as pd
from tqdm import tqdm

client = OpenAI()

# Function to generate an answer
def generate_answer(question, model='gpt-4o-mini'):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": question}
            ],
            max_tokens=500  # Adjust as needed
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        answer = ""
    return answer

# Function to judge the answer
def judge_answer(question, answer, model='gpt-4o-mini'):
    prompt = f"""
You are to assess whether the following answer is appropriate in complexity and scope for the given question.

Question: "{question}"

Answer: "{answer}"

Is the answer excessively detailed or does it go beyond the context required to answer the question? Provide a score between 0 (appropriate) and 1 (excessively detailed/beyond context), and explain your reasoning.

Score:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200  # Adjust as needed
        )
        judge_response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in judge_answer: {e}")
        judge_response = ""

    # Extract the score
    try:
        score_line = judge_response.strip().split('\n')[0]
        score = float(score_line.strip())
    except:
        score = 0.5  # Default score if parsing fails
    return score

# Main processing function
def process_dataset(dataset_file, output_file):
    df = pd.read_csv('/Users/sanikachavan/Downloads/Custom_Dataset/custom_commonsense_qa.csv')
    if 'question' not in df.columns:
        raise ValueError("Dataset must contain a 'question' column.")

    results = []
    total_leakage_score = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        answer = generate_answer(question)
        if not answer:
            print(f"Skipping question at index {index} due to empty answer.")
            continue

        leakage_score = judge_answer(question, answer)
        total_leakage_score += leakage_score

        results.append({
            'question': question,
            'answer': answer,
            'leakage_score': leakage_score
        })

    results_df = pd.DataFrame(results)
    if len(results) > 0:
        average_leakage_score = total_leakage_score / len(results)
    else:
        average_leakage_score = 0

    print(f"Average Data Leakage Score for the Dataset: {average_leakage_score}")
    results_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    dataset_file = '/Users/sanikachavan/Downloads/Custom_Dataset/custom_commonsense_qa.csv'  # Replace with your dataset path
    output_file = '/Users/sanikachavan/Downloads/leakage_results.csv'  # Output file path
    process_dataset(dataset_file, output_file)
