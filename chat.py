import torch
from utils import *
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import random_split

data = torch.load('Reddit/reddit1.pt')
train_loader = torch.load('Reddit/0_10_0/train_sampler1.pt')
val_loader = torch.load('Reddit/0_10_0/val_sampler1.pt')
test_loader = torch.load('Reddit/0_10_0/test_sampler1.pt')

system_instruction = "Don't have extra blank lines and symbols after Answer! "
global_prompt = (
    "You are provided with a list of Reddit users' posts. Each user is classified as either popular or normal "
    "based on their interactions and content."
    )

unique_prompt = (
    "Your task is to generate a brief causal text for each user that "
    "directly relates to classifying them as either popular or normal. Ensure that the generated text is specific "
    "to the user and reflects features that help distinguish them between these two categories.\n"
    "The posts for each user are separated by semicolons. Here is the format for each user's posts:\n"
    "1. [user1's post]\n"
    "2. [user2's post]\n"
    "3. [user3's post]\n"
    "...\n"
    "For each user, generate the causal text as follows:\n"
    "1. [Generated causal text for user1]\n"
    "2. [Generated causal text for user2]\n"
    "3. [Generated causal text for user3]\n"
    "...\n"
    "Remember: Each user's causal text should occupy only one single line."
)
common_prompt = (
    "However, your task is to generate a brief non-causal text for each user that is unrelated to classifying "
    "them as popular or normal. The non-causal text should capture generic or background information that is not"
    "indicative of the user’s classification. Focus on generating text that doesn't directly contribute to "
    "differentiating users based on their categories.\n"
    "The posts for each user are separated by semicolons. Here is the format for each user's posts:\n"
    "1. [user1's post]\n"
    "2. [user2's post]\n"
    "3. [user3's post]\n"
    "...\n"
    "For each user, generate the non-causal text as follows:\n"
    "1. [Generated non-causal text for user1]\n"
    "2. [Generated non-causal text for user2]\n"
    "3. [Generated non-causal text for user3]\n"
    "...\n"
    "Remember: Each user's non-causal text should occupy only one single line."
)
model_name = "gemma-2-9b-it"
#llm = LlamaForCausalLM.from_pretrained(model_name).cuda()
#tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

llm.eval()

def generate_summary(batch, llm, tokenizer, max_retries=1):
    question = "The posts of these users are as follows:\n"
    for i in range(len(batch.subset)):
        if len(data.raw_texts[batch.subset[i]]) > 1200:
            question += f"{i + 1}. [{data.raw_texts[batch.subset[i]][:1200]}]\n"
        else:
            question += f"{i + 1}. [{data.raw_texts[batch.subset[i]]}]\n"
    input_text = f"{system_instruction}\nPrompt: {global_prompt}\n{unique_prompt}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(llm.device)
    print(input_text)
    for attempt in range(max_retries):
        outputs = llm.generate(**inputs, max_new_tokens=550)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if 'Answer:' in answer:
            answer = answer.split('Answer:')[1]
        print(answer)
        results = [line.strip() for line in answer.split('\n') if line.strip()]

        if len(results) == len(batch.subset):
            return results
        else:
            print(f"Format mismatch, retry {attempt + 1}/{max_retries}.")
            feedback_prompt = f"The previous response had formatting errors. Each user's causal text should occupy only one single line. The incorrect response was:\n\n{answer}\n\nPlease correct the format as follows:{system_instruction}\nPrompt: {global_prompt}\n{unique_prompt}\nQuestion: {question}\nCorrected Answer:"
            inputs = tokenizer(feedback_prompt, return_tensors="pt").to(llm.device)
    return None


train = []
for batch in train_loader:
    results = generate_summary(batch, llm, tokenizer)
    if results:
        batch.unique = results
        train.append(batch)
    else:
        train.append(batch)
torch.save(train, 'Reddit/train.pt')

val = []
for batch in val_loader:
    results = generate_summary(batch, llm, tokenizer)
    if results:
        batch.unique = results
        val.append(batch)
    else:
        val.append(batch)
torch.save(val, 'Reddit/val.pt')

test = []
for batch in test_loader:
    results = generate_summary(batch, llm, tokenizer)
    if results:
        batch.unique = results
        test.append(batch)
    else:
        test.append(batch)
torch.save(test, 'Reddit/test.pt')




