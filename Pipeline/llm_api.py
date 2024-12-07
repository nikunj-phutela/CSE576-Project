import openai
from openai import OpenAI
import anthropic
import torch
import transformers
import re

# Initialize OpenAI client
client = OpenAI(api_key="your_openai_api_key_here")

# Initialize Anthropic client
client_cl = anthropic.Anthropic(api_key="your_anthropic_api_key_here")

# Load the Llama model using transformers for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},  # Use bfloat16 for memory efficiency
    device_map="auto"  # Automatically map the model to available devices
)

# Function to interact with Claude (Anthropic) model
def llm_claude(system_prompt, user_prompt, model_name, temp):
    # Send the prompt to the Anthropic API and get the response
    response = client_cl.messages.create(
                model=model_name,  # Model name (e.g., Claude version)
                max_tokens=400,  # Max tokens for the response
                system=system_prompt,  # System instruction
                messages=[
                    {"role": "user", "content": user_prompt}  # User's input prompt
                ],
                temperature=temp  # Sampling temperature
            )
    return response

# Function to judge if a response can be logically entailed from the given context
def judge_response(answer: str, response: str):
    # Define a system prompt for binary judgment (TRUE/FALSE)
    system_prompt = "Your role is to judge a given task. Provide your judgement in just one word, either TRUE or FALSE."
    user_prompt = f"""
    Can the given idea be entailed from the context?
    Idea: {answer}
    Context: {response}
    """
    # Send prompt to OpenAI GPT API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    completions = response.choices[0].message.content.strip()
    # Interpret and return the result as boolean
    if completions == "TRUE":
        return True
    elif completions == "FALSE":
        return False
    else:
        return ""

# Function to interact with GPT-4 (OpenAI)
def llm_gpt4o(system_prompt: str, user_prompt: str, top_k):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,  # Low temperature for deterministic output
        max_tokens=128  # Limit on response length
    )
    return response

# Function to interact with GPT-3.5 (OpenAI)
def llm_gpt3(system_prompt: str, user_prompt: str, top_k):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,  # Low temperature for deterministic output
        n=top_k  # Number of completions to generate
    )
    return response

# Function to interact with Llama model for text generation
def llm_llama(prompt: str, top_k):
    outputs = pipeline(
            prompt,
            num_return_sequences=top_k,  # Number of completions to generate
            max_new_tokens=100,  # Max tokens per completion
            temperature=0.7,  # Sampling temperature for creative output
            do_sample=True,  # Enable sampling
            pad_token_id=pipeline.tokenizer.eos_token_id,  # Padding ID
            return_full_text=False,  # Return only the new text
            top_k=top_k  # Restrict sampling to top-k tokens
        )
    completions = []
    for output in outputs:
        # Clean up the generated text
        completed_question = output['generated_text'].strip()
        completions.append(completed_question)
    return completions

# Extract final answer from a response using regex
def extract_final_answer(response_text):
    match = re.search(r"\*\*([^*]+)\*\*", response_text)  # Match text between ** **
    if match:
        candidate = match.group(1).strip()
        if candidate != "[ANSWER]":
            return candidate  
    second_answer_match = re.search(r"A:\s+(.+)", response_text)  # Match text after "A:"
    if second_answer_match:
        return second_answer_match.group(1).strip()
    return response_text

# Main function to route LLM requests based on the selected model
def use_llm(llm_id: int, system_prompt: str, user_prompt: str, temp=0.1, top_k=1):
    match llm_id:
        case 0:  # GPT-4
            result = llm_gpt4o(system_prompt, user_prompt, top_k)
            if top_k == 5: return result  # Return all completions if multiple requested
            return result.choices[0].message.content.strip()  # Return the first completion
        case 1:  # GPT-3.5
            result = llm_gpt3(system_prompt, user_prompt, top_k)
            if top_k == 5: return result  # Return all completions if multiple requested
            return result.choices[0].message.content.strip()  # Return the first completion
        case 2:  # Claude model (Sonnet version)
            result = llm_claude(system_prompt, user_prompt, "claude-3-sonnet-20240229", temp)
            return ''.join([block.text for block in result.content])
        case 3:  # Claude 3.5 model
            result = llm_claude(system_prompt, user_prompt, "claude-3-5-sonnet-20241022", temp)
            return ''.join([block.text for block in result.content])
        case 4:  # Llama model
            prompt = system_prompt + user_prompt
            result = llm_llama(prompt, top_k)
            if top_k == 5: return result  # Return all completions if multiple requested
            result = result[0]
            ans = extract_final_answer(result)  # Extract and return the final answer
            return ans
