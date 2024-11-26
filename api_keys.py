import openai
from openai import OpenAI
import anthropic
import torch
import transformers

client = OpenAI(api_key = "")

client_cl = anthropic.Anthropic(api_key="")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

def generate_relevant_info_gpt4o(key_terms: list):
  terms = ", ".join(key_terms)

  system_prompt = "You are a helpful assistant that gives relevant information. Provide a comprehensive context or paragraph that meaningfully incorporates the given key terms or words."
  user_prompt = f"""
    Provide relevant information and context incorporating the following key terms.
    Key terms: {terms}
    """

  response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

  completions = response.choices[0].message.content.strip()

  return completions


def seenQ_claude_gsm8k(data):
    result = {}
    for index, row in data[:].iterrows():
        prompt = f"Complete the sentence according to the hint in one word: [{row['masked_question']}]. Reply the answer only in one word without full sentence."
        print(f"{index}: {prompt}")
        response = client_cl.messages.create(
            model="claude-3-sonnet-20240229", 
            max_tokens=128,                    
            messages=[
                {"role": "user", "content": prompt}  
            ]
        )

        response_text = ''.join([block.text for block in response.content])

        result[row['question']] = response_text
    return result

def seenQ_gpt_gsm8k(data):
    result = {}
    for index, row in data[:].iterrows():
        prompt = f"Complete the sentence in one word by replacing the word in (): [{row['masked_question']}]. Reply the answer only in one word without full sentence."
        print(str(index)+": "+ prompt)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content":prompt}
            ],
            temperature = 0.1,
            max_tokens = 128
        )
    response = completion["choices"][0]["message"]["content"].replace('\n', ' ')
    result[row['question']] = response
    return result
  
def seenQ_llama(data):
    result = {}
    for index, row in data[:].iterrows():
        url = row['url']
        if isinstance(url, str):
            url = url.replace(row['answer'], "")
            url = url.replace(row['answer'].lower(), "")
            url = url.replace(row['answer'].capitalize(), "")
        
        prompt = f"Complete the sentence according to the hint in one word: [{row['question']}]. Hint:[{url}]. Reply with ONLY ONE word, not with the entire sentence."
        print(prompt)
        response = pipeline(prompt, max_new_tokens=1)
        response_text = response[0]['generated_text'].strip()
        result[row['question']] = response_text
    return result


def seenQA_claude_commonsenseqa(data):
    result = {}
    for index, row in data[:].iterrows():
        prompt = f"Complete the incomplete option in the answer to the question in one word: [{row['question_answer']}]. \nReply the answer only, the option without the full sentence."
        response = client_cl.messages.create(
            model="claude-3-sonnet-20240229", 
            max_tokens=128,                    
            messages=[
                {"role": "user", "content": prompt}  
            ]
        )
        response_text = ''.join([block.text for block in response.content])

        result[row['question']] = response_text
    return result

def seenQA_gpt_commonsenseqa(data):
    result = {}
    for index, row in data[:].iterrows():
        prompt = f"Complete the incomplete option in the answer to the question in one word: [{row['question_answer']}]. \nReply the answer only, the option without the full sentence."
        completion = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "user", "content":prompt}
            ],
            temperature = 0.1,
            max_tokens = 128
        )
        response = completion["choices"][0]["message"]["content"].replace('\n', ' ')
        result[row['question']] = response
    return result


def seenQA_claude_mmlu(data):
    result = {}
    for index, row in data[:].iterrows():
        prompt = f"Complete the incomplete option in the answer to the question in one word: [{row['question_answer']}]. \nReply the answer only, the option without the full sentence."
        response = client_cl.messages.create(
            model="claude-3-5-sonnet-20241022", 
            max_tokens=128,                    
            messages=[
                {"role": "user", "content": prompt}  
            ]
        )
        response_text = ''.join([block.text for block in response.content])
        result[row['question']] = response_text
    return result


def seenQA_gpt_mmlu(data):
    result = {}
    for index, row in data[:].iterrows():
        prompt = f"Complete the incomplete option in the answer to the question in one word: [{row['question_answer']}]. \nReply the answer only, the option without the full sentence."
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content":prompt}
            ],
            temperature = 0.1,
            max_tokens = 128
        )
        response = completion["choices"][0]["message"]["content"].replace('\n', ' ')
        result[row['question']] = response
    return result


  
  

  


