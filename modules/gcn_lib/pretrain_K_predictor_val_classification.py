import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Flickr
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split

class VAELikeKPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_classes=21):
        super(VAELikeKPredictor, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)
        self.fc_classify = nn.Linear(latent_dim, num_classes) 

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.xavier_normal_(self.fc_logvar.weight)
        nn.init.xavier_normal_(self.fc_classify.weight)
        nn.init.constant_(self.fc_classify.bias, 0.0)

    def forward(self, x):
        # x has shape (N, C)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = self.fc_classify(z)
        return logits, mu, logvar

    def loss_function(self, logits, mu, logvar, true_K):
        classification_loss = F.cross_entropy(logits, true_K)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        return classification_loss + kl_loss

    def accuracy(self, pred, true):
        return (pred == true).float().mean().item()


def pretrain_vaelike_kpredictor():
    dataset = Flickr(root='data/Flickr', transform=NormalizeFeatures())
    data = dataset[0]

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print()
    print(data)
    print('====================')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    

    in_channels = dataset.num_features
    hidden_channels = 64
    latent_dim = 32
    num_classes = 21
    
    model = VAELikeKPredictor(in_channels, hidden_channels, latent_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    true_K = torch.clamp((data.edge_index[0].bincount(minlength=data.num_nodes)), max=20).long()

    train_idx, val_idx = train_test_split(range(data.num_nodes), test_size=0.2, random_state=0)
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)

    model.train()
    num_epochs = 1201
    best_loss = float('inf')
    save_path = 'vaelike_kpredictor_flickr_best_val.pth'
    log_file_path = 'k_predictor_training_log_val.txt'
    with open(log_file_path, 'w') as log_file:

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            logits, mu, logvar = model(data.x[train_idx])
            pred_K = logits.argmax(dim=1)
            loss = model.loss_function(logits, mu, logvar, true_K[train_idx])
            train_accuracy = model.accuracy(pred_K, true_K[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits, val_mu, val_logvar = model(data.x[val_idx])
                val_pred_K = val_logits.argmax(dim=1)
                val_loss = model.loss_function(val_logits, val_mu, val_logvar, true_K[val_idx])
                accuracy = model.accuracy(val_pred_K, true_K[val_idx])

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss.item():.4f},Train Acc: {train_accuracy:.8f},  Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy:.8f}')
                log_file.write(f'Epoch {epoch}, Train Loss: {loss.item():.4f},Train Acc: {train_accuracy:.8f},  Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy:.8f}\n')

            if val_loss.item() < 700 and val_loss.item() < best_loss:
                best_loss = val_loss.item()
                acc_best_loss = accuracy
                torch.save(model.state_dict(), save_path)
                print(f'Best model saved at epoch {epoch} with val loss {best_loss:.4f} and acc {acc_best_loss:.8f}')
                log_file.write(f'Best model saved at epoch {epoch} with val loss {best_loss:.4f} and acc {acc_best_loss:.8f}\n')

        print(f'Training completed. Best model saved with val loss {best_loss:.4f} at {save_path} and acc {acc_best_loss:.8f}')
        log_file.write(f'Training completed. Best model saved with val loss {best_loss:.4f} at {save_path} and acc {acc_best_loss:.8f}\n')

def load_pretrained_vaelike_kpredictor(model, weight_path='vaelike_kpredictor_flickr_best_val.pth'):
    model.load_state_dict(torch.load(weight_path))
    return model

if __name__ == "__main__":
    pretrain_vaelike_kpredictor()
    
    # Load pretrained model weights into a new instance
    # model = VAELikeKPredictor(in_channels=10, hidden_channels=64, latent_dim=32, num_classes=21)
    # model = load_pretrained_vaelike_kpredictor(model)
    # Now, model can be used in your current training process
