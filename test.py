import torch
import random
import numpy as np
import argparse
from torch.utils.data import random_split
from models import GCN, DualGNN, GraphSAGE, GAT
from geniepath import GeniePathLazy, GeniePath
from dga import DGA
from bwgnn import BWGNN
from caregnn import CAREGNN
from pmp import LASAGE_S
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import random_split
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
from utils import FocalLoss, visualization, ECELoss
import statistics
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--path', type=str, default="Reddit/0_10_0/")
args = parser.parse_args()

# criterion = FocalLoss(alpha=0.5, gamma=2).cuda()
data = torch.load('Reddit/reddit1.pt').cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
train_loader = torch.load(args.path + "train_loader1.pt")
random.shuffle(train_loader)
val_loader = torch.load(args.path + "val_loader1.pt")
test_loader = torch.load(args.path + "test_loader1.pt")
# encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = torch.load('Reddit/embeddings1.pt')
embeddings = torch.Tensor(embeddings).cuda()
accumulation_steps = 10
label0, label1 = [], []
for batch in train_loader:
    if data.y[batch.central] == 0:
        label0.append(batch.central)
    else:
        label1.append(batch.central)


def train_gnn(model, optimizer, train_loader):
    model.train()
    all_labels = []
    all_predictions = []
    all_pred = []
    all_emb = []
    total_loss = total_correct = total_examples = 0
    batch_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        # text_embeddings = torch.cat((text_embeddings, batch.unique_embeddings), dim=1)
        if hasattr(batch, "unique_embeddings"):
            text_embeddings = batch.unique_embeddings
        else:
            text_embeddings = embeddings[batch.subset]
        edge_index = batch.edge_index
        subset = batch.subset
        '''
        if len(batch.subset) == 10:
            # 0 1700
            if data.y[batch.central] == 0:
                text_embeddings = torch.cat((text_embeddings, embeddings[1700].unsqueeze(0)), dim=0)
            else:
                text_embeddings = torch.cat((text_embeddings, embeddings[0].unsqueeze(0)), dim=0)
            edge_index = torch.cat((edge_index, torch.LongTensor([[0, 10], [10, 0]]).cuda()), dim=1)
        '''
        emb32, output = model(data.x[batch.subset].cuda(), edge_index.cuda())
        emb32 = emb32[subset == batch.central][0]
        output = output[subset == batch.central][0]
        all_emb.append(output.detach().cpu().numpy())
        pred = output.argmax(dim=-1)
        all_pred.append(pred.cpu().numpy())
        label = data.y[batch.central]
        all_labels.append(label.cpu().item())  # True labels
        all_predictions.append(F.sigmoid(output).detach().cpu().numpy())  # Model's predicted probabilities
        loss = criterion(output, label.cuda())
        total_loss += float(loss) * 1
        total_correct += int((output.argmax(dim=-1) == label.cuda()).sum())
        total_examples += 1
        batch_loss += loss
        if (i + 1) % accumulation_steps == 0 and batch_loss != 0:
            batch_loss.backward()
            optimizer.step()  # 进行一次权重更新
            optimizer.zero_grad()  # 重置梯度
            batch_loss = 0
    if batch_loss != 0:
        batch_loss.backward()
        optimizer.step()
    torch.save(all_emb, args.path + 'all_emb.pt')
    torch.save(all_labels, args.path + 'all_label.pt')
    f1_macro = f1_score(np.array(all_labels), np.array(all_pred), average='macro')
    roc_auc = roc_auc_score(np.array(all_labels), np.array(all_predictions)[:, 1])
    ece = ECELoss(torch.Tensor(np.array(all_emb)), torch.LongTensor(np.array(all_labels)))
    print(roc_auc, f1_macro, ece,total_correct / total_examples, total_loss / total_examples)


@torch.no_grad()
def test_gnn(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    all_pred = []
    all_emb = []
    total_loss = total_correct = total_examples = 0
    for batch in test_loader:
        # text_embeddings = torch.cat((text_embeddings, batch.unique_embeddings), dim=1)
        if hasattr(batch, "unique_embeddings"):
            text_embeddings = batch.unique_embeddings
        else:
            text_embeddings = embeddings[batch.subset]
        edge_index = batch.edge_index
        subset = batch.subset
        '''
        if len(batch.subset) == 10:
            # 0 1700
            count = 3
            for i in range(1, len(batch.subset)):
                if data.y[batch.subset[i]] == data.y[batch.central]:
                    if data.y[batch.central] == 0:
                        index = randint(0, len(label1) - 1)
                        text_embeddings[i] = embeddings[label1[index]]
                        subset[i] = torch.LongTensor([label1[index]]).cuda()
                    else:
                        index = randint(0, len(label0) - 1)
                        text_embeddings[i] = embeddings[label0[index]]
                        subset[i] = torch.LongTensor([label0[index]]).cuda()
                    #count -= 1
                    #if count == 0:
                        #break

            if data.y[batch.central] == 0:   
                text_embeddings = torch.cat((text_embeddings, embeddings[0].unsqueeze(0)), dim=0)
            else:
                text_embeddings = torch.cat((text_embeddings, embeddings[1700].unsqueeze(0)), dim=0)
            edge_index = torch.cat((edge_index, torch.LongTensor([[0, 10, 10], [10, 0, 10]]).cuda()), dim=1)
            '''
        emb32, output = model(data.x[batch.subset].cuda(), edge_index.cuda())
        emb32 = emb32[subset == batch.central][0]
        output = output[subset == batch.central][0]
        all_emb.append(output.cpu().numpy())
        pred = output.argmax(dim=-1)
        all_pred.append(pred.cpu().numpy())
        label = data.y[batch.central]
        all_labels.append(label.cpu().item())  # True labels
        all_predictions.append(F.sigmoid(output).cpu().numpy())  # Model's predicted probabilities
        loss = criterion(output, label.cuda())
        total_loss += float(loss) * 1
        total_correct += int((output.argmax(dim=-1) == label.cuda()).sum())
        total_examples += 1
    #torch.save(all_emb, args.path + 'all_emb.pt')
    #torch.save(all_labels, args.path + 'all_label.pt')
    torch.save(all_predictions, args.path + 'all_pred.pt')
    f1_macro = f1_score(np.array(all_labels), np.array(all_pred), average='macro')
    roc_auc = roc_auc_score(np.array(all_labels), np.array(all_predictions)[:, 1])
    ece = ECELoss(torch.Tensor(np.array(all_emb)), torch.LongTensor(np.array(all_labels)))
    print(roc_auc, f1_macro, ece, total_correct / total_examples, total_loss / total_examples)

    return total_correct / total_examples, total_loss / total_examples, roc_auc, f1_macro, ece


def main_gnn(model, optimizer):
    best = 0

    for epoch in range(args.epochs):
        train_gnn(model, optimizer, train_loader)
        acc_val, loss_val, roc_auc, f1_macro, ece = test_gnn(model, val_loader)
        if f1_macro > best:
            torch.save(model.state_dict(), args.path + 'gnn.pth')
            best = f1_macro

    state_dict = torch.load(args.path + 'gnn.pth')
    model.load_state_dict(state_dict)

    acc, loss, roc_auc, f1_macro, ece = test_gnn(model, test_loader)

    return acc, loss, roc_auc, f1_macro, ece


acc_final, auc_final, auc_homo, auc_hetero, f1_final, ece_final = [], [], [], [], [], []
for i in range(5):
    torch.manual_seed(i)
    random.shuffle(train_loader)
    for j in range(1):
        # 4096 384
        #gnn_model = GCN(4096, args.hidden, 2).cuda()
        gnn_model = GeniePathLazy(4096, 2, 'cuda').cuda()
        #gnn_model = DualGNN(2, gnn).cuda()
        optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)
        acc, loss, roc_auc, f1_macro, ece = main_gnn(gnn_model, optimizer_gnn)
        acc_final.append(acc), auc_final.append(roc_auc), f1_final.append(f1_macro), ece_final.append(ece)
        all_pred = torch.Tensor(torch.load(args.path + 'all_pred.pt'))
        all_label = torch.LongTensor(torch.load(args.path + 'all_label.pt'))
        # homo_auc = roc_auc_score(np.array(all_label[data.homo.cpu()]), np.array(all_pred[data.homo.cpu()])[:, 1])
        # hetero_auc = roc_auc_score(np.array(all_label[data.hetero.cpu()]), np.array(all_pred[data.hetero.cpu()])[:, 1])
        # print(homo_auc, hetero_auc)
        homo_auc, hetero_auc = 0, 0
        auc_homo.append(homo_auc), auc_hetero.append(hetero_auc)
print(f'acc: {statistics.mean(acc_final):2f}±{statistics.stdev(acc_final):2f}',
      f'auc: {statistics.mean(auc_final):2f}±{statistics.stdev(auc_final):2f}',
      f'homo_auc: {statistics.mean(auc_homo):2f}±{statistics.stdev(auc_homo):2f}',
      f'hetero_auc: {statistics.mean(auc_hetero):2f}±{statistics.stdev(auc_hetero):2f}',
      f'f1: {statistics.mean(f1_final):2f}±{statistics.stdev(f1_final):2f}',
      f'ece: {statistics.mean(ece_final):2f}±{statistics.stdev(ece_final):2f}',
      )
