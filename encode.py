import torch
import argparse
from torch.utils.data import random_split
from models import GCN
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import random_split

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--path', type=str, default="Instagram/")
args = parser.parse_args()
criterion = torch.nn.CrossEntropyLoss().cuda()

data = torch.load('Instagram/instagram.pt')
train_loader = torch.load(args.path + "train.pt")
val_loader = torch.load(args.path + "val.pt")
test_loader = torch.load(args.path + "test.pt")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

train = []
for batch in train_loader:
    text = [data.raw_texts[i] for i in batch.subset]
    if hasattr(batch, "unique"):
        text = batch.unique
        for i in range(len(text)):
            text[i] = text[i][2:].strip()
    unique_embeddings = encoder.encode(text)
    unique_embeddings = torch.Tensor(unique_embeddings).cuda()
    batch.unique_embeddings = unique_embeddings
    train.append(batch)
torch.save(train, args.path + "train.pt")

val = []
for batch in val_loader:
    text = [data.raw_texts[i] for i in batch.subset]
    if hasattr(batch, "unique"):
        text = batch.unique
        for i in range(len(text)):
            text[i] = text[i][3:].strip()
    unique_embeddings = encoder.encode(text)
    unique_embeddings = torch.Tensor(unique_embeddings).cuda()
    batch.unique_embeddings = unique_embeddings
    val.append(batch)
torch.save(val, args.path + "val.pt")

test = []
for batch in test_loader:
    text = [data.raw_texts[i] for i in batch.subset]
    if hasattr(batch, "unique"):
        text = batch.unique
        for i in range(len(text)):
            text[i] = text[i][3:].strip()
    unique_embeddings = encoder.encode(text)
    unique_embeddings = torch.Tensor(unique_embeddings).cuda()
    batch.unique_embeddings = unique_embeddings
    test.append(batch)
torch.save(test, args.path + "test.pt")
