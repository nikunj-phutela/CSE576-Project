import pandas as pd
from tqdm import tqdm
import relevant_info as rel_inf
import similar_q as sim_q
import seen_q_n_a as s_q_a
import seen_q as s_q

# Mapping of models and datasets for user selection
models = {0: "GPT-4o", 1: "GPT-3.5", 2: "Claude 3", 3: "Claude 3.5", 4: "llama 3.2"}
dataset = {0: "TruthfulQA", 1: "CommonsenseQA", 2: "MMLU", 3: "GSM8K", 4: "All"}

# Enable progress bar for DataFrame operations
tqdm.pandas()

# User chooses the dataset to process
print(dataset, "Choose the dataset")
dataset_id = int(input("(Input dataset ID): "))

# Load the corresponding dataset
match dataset_id:
    case 0:
        df_file = pd.read_csv('Data/TruthfulQA.csv')
    case 1:
        df_file = pd.read_csv('Data/commonsense_pass_pipeline.csv')
    case 2:
        df_file = pd.read_csv('Data/mmlu_pass_pipeline.csv')
    case 3:
        df_file = pd.read_csv('Data/gsm8k.csv')

# Initialize the output DataFrame with required columns
df_output = df_file.copy()
df_output['extracted_terms'] = None
df_output['rel_info_generated'] = None
df_output['leakage_in_bucket'] = None
df_output['can_be_data_leakage'] = False

# User chooses the LLM model to evaluate
print(models, "Choose the llm model")
llm_id = int(input("(Input model ID): "))

# Define the pipeline function for data processing
def run_pipeline(row):
    global llm_id
    global dataset_id
    try:
        # Check for Seen Q&A leakage
        if ((dataset_id not in [0, 3]) and s_q_a.seen_q_n_a(row, llm_id)[1]):
            row['leakage_in_bucket'] = "Seen Q & A"
            row['can_be_data_leakage'] = True
            return row
        # Check for Seen Question leakage
        elif s_q.seen_q(row, llm_id, dataset_id)[1]:
            row['leakage_in_bucket'] = "Seen Q"
            row['can_be_data_leakage'] = True
            return row
        # Check for Similar Question leakage
        elif sim_q.check_similar_q(row, llm_id)[1]:
            row['leakage_in_bucket'] = "Similar Q"
            row['can_be_data_leakage'] = True
            return row
        # Check for Relevant Information leakage
        elif rel_inf.relevant_info(row, llm_id)[1]:
            row['leakage_in_bucket'] = "Relevant Info"
            row['can_be_data_leakage'] = True
            return row
    except Exception as e:
        # Handle unexpected errors gracefully
        print(f"Skipping row due to unexpected error: {e}")
        return row
    return row

# Apply the pipeline function to a subset of the data for faster testing
df_output = df_output[:100]
df_output = df_output.progress_apply(run_pipeline, axis=1)

# Save the processed output to a CSV file
df_output.to_csv(f'Outputs/output_{llm_id}_{dataset_id}.csv', index=False)
