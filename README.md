# Project CSE 576 - Data Leakage Detection in LLMs

## Project Overview
This project focuses on detecting **data leakage** in large language models (LLMs) through a systematic pipeline. 

### Consists of 4 Stages:
1. **Seen Question**
2. **Seen Question and Answer**
3. **Seen Similar Question**
4. **Seen Relevant Information**

We performed stage-wise analysis, and the details for each stage are available in their respective folders.

## Pipeline
The **main Data Leakage Pipeline** is located in the `Pipeline` folder. It includes all four stages and provides runnable code to analyze data leakage in:

### Supported Models
- **GPT-4o**
- **GPT-3.5**
- **Claude 3.5**
- **Claude 3**
- **Llama 3.2 Instruct**

### Supported Datasets
- **TruthfulQA**
- **CommonsenseQA**
- **MMLU**
- **GSM8K**

## Things to Do Before Running the Pipeline
Before you can run the pipeline, ensure the following steps are completed:

1. **Python Version**:
   - Ensure you have **Python 3.8** or above installed.  
     You can check your Python version by running:
     ```bash
     python3 --version
     ```

2. **Install Dependencies**:
   - Use the `requirements.txt` file to install all necessary libraries and dependencies.
   - Run the following command:
     ```bash
     pip3 install -r requirements.txt
     ```

3. **Set Up API Keys**:
   - The pipeline requires API keys for **Anthropic** and **OpenAI**.
   - Add your keys to the `llm_api.py` file:
    ```python
    client = OpenAI(api_key="your_openai_api_key_here")
    ```
    ```python
    client_cl = anthropic.Anthropic(api_key="your_anthropic_api_key_here")
    ```
   - Replace `"your_anthropic_api_key_here"` and `"your_openai_api_key_here"` with your actual API keys.

4. **Verify Data**:
   - Ensure the required datasets (`TruthfulQA`, `CommonsenseQA`, `MMLU`, and `GSM8K`) are available and correctly formatted in the `Data/` folder.  

Once these steps are completed, you are ready to run the pipeline.


### How to Run the Pipeline
1. Navigate to the `Pipeline` folder.
2. Run the `pipeline.py` file:
   ```bash
   python3 pipeline.py
