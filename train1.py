from peft import LoraConfig, TaskType
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *
import torch
import numpy as np
import argparse
from torch.utils.data import random_split
from models import GCN
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import random_split
from sklearn.metrics import f1_score, roc_auc_score
import random

parser = argparse.ArgumentParser()
parser.add_argument('--outer_epochs', type=int, default=3,
                    help='Number of epochs to train.')
parser.add_argument('--inner_epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.1)

args = parser.parse_args()
criterion = torch.nn.CrossEntropyLoss().cuda()

lora_config = LoraConfig(
    r=8,  # LoRA的秩 (rank)
    lora_alpha=32,  # LoRA scaling factor
    target_modules=["q_proj", "v_proj"],  # 定义在哪些层应用LoRA (可根据具体模型调整)
    lora_dropout=0.1,  # dropout概率
    bias="none"  # 不调整模型中的bias项
)
model_name = "gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
model = get_peft_model(llm, lora_config)
model.print_trainable_parameters()

data = torch.load('Instagram/instagram.pt').cuda()
train_loader = torch.load('Instagram/0_10_0/train_sampler.pt')
random.shuffle(train_loader)
val_loader = torch.load('Instagram/0_10_0/val_sampler.pt')
test_loader = torch.load('Instagram/0_10_0/test_sampler.pt')

encoder = SentenceTransformer("all-MiniLM-L6-v2")
gnn_model = GCN(384, args.hidden, 2, args.dropout).cuda()
state_dict = torch.load('Instagram/model_lora/gnn.pth')
gnn_model.load_state_dict(state_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)
accumulation_steps = 10

system_instruction = "Don't have extra blank lines and symbols after Answer! "
global_prompt = (
    "You are provided with a list of Instagram users' personal introductions. Each user is classified as either commercial or normal "
    "based on their profile characteristics."
)

unique_prompt = (
    "Your task is to generate a brief causal text for each user that "
    "highlights their unique characteristics without predicting their classification as commercial or normal. "
    "Focus on analyzing the user's profile to extract relevant features that can be used to generate causal text.\n"
    "The introductions for each user are separated by semicolons. Here is the format for each user's introduction:\n"
    "1. [user1's introduction]\n"
    "2. [user2's introduction]\n"
    "3. [user3's introduction]\n"
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
    "them as commercial or normal. The non-causal text should capture generic or background information that is not "
    "indicative of the user’s classification. Focus on generating text that doesn't directly contribute to "
    "differentiating users based on their categories.\n"
    "The introductions for each user are separated by semicolons. Here is the format for each user's introduction:\n"
    "1. [user1's introduction]\n"
    "2. [user2's introduction]\n"
    "3. [user3's introduction]\n"
    "...\n"
    "For each user, generate the non-causal text as follows:\n"
    "1. [Generated non-causal text for user1]\n"
    "2. [Generated non-causal text for user2]\n"
    "3. [Generated non-causal text for user3]\n"
    "...\n"
    "Remember: Each user's non-causal text should occupy only one single line."
)


def train_model(model, gnn_model, optimizer, train_loader):
    model.train()
    gnn_model.eval()
    batch_loss = 0
    train = []
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        question = "The introductions of these users are as follows:\n"
        for i in range(len(batch.subset)):
            if len(data.raw_texts[batch.subset[i]]) > 1200:
                question += f"{i + 1}. [{data.raw_texts[batch.subset[i]][:1200]}]\n"
            else:
                question += f"{i + 1}. [{data.raw_texts[batch.subset[i]]}]\n"
        print(question)
        unique_text = f"{system_instruction}\nPrompt: {global_prompt}\n{unique_prompt}\nQuestion: {question}\nAnswer:"
        unique_input = tokenizer(unique_text, return_tensors="pt").to(model.device)
        unique_output = model.generate(**unique_input, max_new_tokens=550)
        unique_answer = tokenizer.decode(unique_output[0], skip_special_tokens=True)
        unique_answer = unique_answer.split('Answer:')[1]
        print(unique_answer)
        unique_result = [line.strip() for line in unique_answer.split('\n') if line.strip()]
        if len(unique_result) != len(batch.subset):
            train.append(batch)
            continue
        common_text = f"{system_instruction}\nPrompt: {global_prompt}\n{common_prompt}\nQuestion: {question}\nAnswer:"
        common_input = tokenizer(common_text, return_tensors="pt").to(model.device)
        common_output = model.generate(**common_input, max_new_tokens=550)
        common_answer = tokenizer.decode(common_output[0], skip_special_tokens=True)
        common_answer = common_answer.split('Answer:')[1]
        print(common_answer)
        common_result = [line.strip() for line in common_answer.split('\n') if line.strip()]
        if len(common_result) != len(batch.subset):
            train.append(batch)
            continue
        unique_embeddings = encoder.encode(unique_result)
        unique_embeddings = torch.Tensor(unique_embeddings).cuda()
        common_embeddings = encoder.encode(common_result)
        common_embeddings = torch.Tensor(common_embeddings).cuda()
        batch.unique = unique_result
        batch.common = common_result
        batch.unique_embeddings = unique_embeddings
        batch.common_embeddings = common_embeddings
        train.append(batch)
        unique_embeddings = gnn_model(unique_embeddings, batch.edge_index.cuda())
        common_embeddings = gnn_model(common_embeddings, batch.edge_index.cuda())
        unique_embeddings = unique_embeddings[batch.subset == batch.central][0]
        unique_label = data.y[batch.central]
        common_embeddings = common_embeddings[batch.subset == batch.central][0]
        loss = causal_loss(unique_embeddings, unique_label.cuda()) + \
               args.alpha * non_causal_loss(common_embeddings) + \
               args.beta * orthogonal_loss(unique_embeddings, common_embeddings)
        batch_loss += loss
        if (i + 1) % accumulation_steps == 0:
            batch_loss.backward()
            optimizer.step()  # 进行一次权重更新
            optimizer.zero_grad()  # 重置梯度
            batch_loss = 0
    if batch_loss != 0:
        batch_loss.backward()
        optimizer.step()
    return train

def train_gnn(gnn_model, gnn_optimizer, train):
    gnn_model.train()
    batch_loss = 0
    gnn_optimizer.zero_grad()
    for i, batch in enumerate(train):
        if hasattr(batch, "unique"):
            unique_embeddings = gnn_model(batch.unique_embeddings, batch.edge_index.cuda())
            common_embeddings = gnn_model(batch.common_embeddings, batch.edge_index.cuda())
            unique_embeddings = unique_embeddings[batch.subset == batch.central][0]
            unique_label = data.y[batch.central]
            common_embeddings = common_embeddings[batch.subset == batch.central][0]
            loss = causal_loss(unique_embeddings, unique_label.cuda()) + \
                   args.alpha * non_causal_loss(common_embeddings) + \
                   args.beta * orthogonal_loss(unique_embeddings, common_embeddings)
            batch_loss += loss
            if (i + 1) % accumulation_steps == 0:
                batch_loss.backward()
                gnn_optimizer.step()  # 进行一次权重更新
                gnn_optimizer.zero_grad()  # 重置梯度
                batch_loss = 0
    if batch_loss != 0:
        batch_loss.backward()
        optimizer.step()

def val_model(model, gnn_model, val_loader):
    model.eval()
    gnn_model.eval()
    val = []
    all_labels = []
    all_predictions = []
    total_loss = total_correct = total_examples = 0
    for i, batch in enumerate(val_loader):
        question = "The introductions of these users are as follows:\n"
        for i in range(len(batch.subset)):
            if len(data.raw_texts[batch.subset[i]]) > 1200:
                question += f"{i + 1}. [{data.raw_texts[batch.subset[i]][:1200]}]\n"
            else:
                question += f"{i + 1}. [{data.raw_texts[batch.subset[i]]}]\n"
        print(question)
        unique_text = f"{system_instruction}\nPrompt: {global_prompt}\n{unique_prompt}\nQuestion: {question}\nAnswer:"
        unique_input = tokenizer(unique_text, return_tensors="pt").to(model.device)
        unique_output = model.generate(**unique_input, max_new_tokens=550)
        unique_answer = tokenizer.decode(unique_output[0], skip_special_tokens=True)
        unique_answer = unique_answer.split('Answer:')[1]
        print(unique_answer)
        unique_result = [line.strip() for line in unique_answer.split('\n') if line.strip()]
        if len(unique_result) != len(batch.subset):
            val.append(batch)
            continue
        common_text = f"{system_instruction}\nPrompt: {global_prompt}\n{common_prompt}\nQuestion: {question}\nAnswer:"
        common_input = tokenizer(common_text, return_tensors="pt").to(model.device)
        common_output = model.generate(**common_input, max_new_tokens=550)
        common_answer = tokenizer.decode(common_output[0], skip_special_tokens=True)
        common_answer = common_answer.split('Answer:')[1]
        print(common_answer)
        common_result = [line.strip() for line in common_answer.split('\n') if line.strip()]
        if len(common_result) != len(batch.subset):
            val.append(batch)
            continue
        unique_embeddings = encoder.encode(unique_result)
        unique_embeddings = torch.Tensor(unique_embeddings).cuda()
        common_embeddings = encoder.encode(common_result)
        common_embeddings = torch.Tensor(common_embeddings).cuda()
        batch.unique = unique_result
        batch.common = common_result
        batch.unique_embeddings = unique_embeddings
        batch.common_embeddings = common_embeddings
        val.append(batch)
        unique_embeddings = gnn_model(unique_embeddings, batch.edge_index.cuda())
        common_embeddings = gnn_model(common_embeddings, batch.edge_index.cuda())
        unique_embeddings = unique_embeddings[batch.subset == batch.central][0]
        unique_label = data.y[batch.central]
        common_embeddings = common_embeddings[batch.subset == batch.central][0]
        loss = causal_loss(unique_embeddings, unique_label.cuda()) + \
               args.alpha * non_causal_loss(common_embeddings) + \
               args.beta * orthogonal_loss(unique_embeddings, common_embeddings)
        all_labels.append(unique_label.cpu().item())  # True labels
        all_predictions.append(F.sigmoid(unique_embeddings).detach().cpu().numpy())  # Model's predicted probabilities
        total_loss += float(loss.detach().cpu().numpy()) * 1
        total_correct += int((unique_embeddings.argmax(dim=-1) == unique_label.cuda()).sum())
        total_examples += 1
    print(all_labels, all_predictions)
    roc_auc = roc_auc_score(np.array(all_labels), np.array(all_predictions)[:, 1])
    print(roc_auc, total_correct / total_examples, total_loss / total_examples)

    return val, total_correct / total_examples, total_loss / total_examples, roc_auc

def val_gnn(gnn_model, val):
    gnn_model.eval()
    all_labels = []
    all_predictions = []
    total_loss = total_correct = total_examples = 0
    for batch in val:
        if hasattr(batch, "unique"):
            unique_embeddings = gnn_model(batch.unique_embeddings, batch.edge_index.cuda())
            common_embeddings = gnn_model(batch.common_embeddings, batch.edge_index.cuda())
            unique_embeddings = unique_embeddings[batch.subset == batch.central][0]
            unique_label = data.y[batch.central]
            common_embeddings = common_embeddings[batch.subset == batch.central][0]
            loss = causal_loss(unique_embeddings, unique_label.cuda()) + \
                   args.alpha * non_causal_loss(common_embeddings) + \
                   args.beta * orthogonal_loss(unique_embeddings, common_embeddings)
            all_labels.append(unique_label.cpu().item())  # True labels
            all_predictions.append(F.sigmoid(unique_embeddings).detach().cpu().numpy())  # Model's predicted probabilities
            total_loss += float(loss.detach().cpu().numpy()) * 1
            total_correct += int((unique_embeddings.argmax(dim=-1) == unique_label.cuda()).sum())
            total_examples += 1

    roc_auc = roc_auc_score(np.array(all_labels), np.array(all_predictions)[:, 1])
    print(roc_auc, total_correct / total_examples, total_loss / total_examples)

    return total_correct / total_examples, total_loss / total_examples, roc_auc

def main(model, gnn_model, optimizer, gnn_optimizer, train_loader, val_loader, test_loader):
    outer_best = -100
    for epoch in range(args.outer_epochs):
        train = train_model(model, gnn_model, optimizer, train_loader)
        val, acc_val, loss_val, roc_auc = val_model(model, gnn_model, val_loader)
        if acc_val + roc_auc - loss_val > outer_best:
            model.save_pretrained('Instagram/model_lora')
            torch.save(train, 'Instagram/model_lora/train.pt')
            torch.save(val, 'Instagram/model_lora/val.pt')
            outer_best = acc_val + roc_auc - loss_val

        inner_best = -100
        for epoch in range(args.inner_epochs):
            train_gnn(gnn_model, gnn_optimizer, train)
            acc_val, loss_val, roc_auc = val_gnn(gnn_model, val)
            if acc_val + roc_auc - loss_val > inner_best:
                torch.save(gnn_model.state_dict(), 'Instagram/model_lora/gnn.pth')
                inner_best = acc_val + roc_auc - loss_val
        state_dict = torch.load('Instagram/model_lora/gnn.pth')
        gnn_model.load_state_dict(state_dict)


main(model, gnn_model, optimizer, gnn_optimizer, train_loader, val_loader, test_loader)
