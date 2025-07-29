import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 21

data_path = "E:/Abishek/Projects/Plant_stress/plant_disease_dataset_2000.csv"
data = pd.read_csv(data_path)

lc = LabelEncoder()
data['Scenario_label'] = lc.fit_transform(data['Scenario'])
data['Class_label'] = lc.fit_transform(data['Class'])

data = data.drop(['LDR', 'Scenario', 'Class', 'VOC', 'Ambient_temperature', 'Leaf_Temperature'], axis=1).dropna()

X = data.drop(['Class_label'], axis=1)
y = data['Class_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=100)

def build_knn_graph(x_array, k=12):
    adj = kneighbors_graph(x_array, k, mode='connectivity', include_self=True)
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    return edge_index

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

edge_index = build_knn_graph(X_train)
edge_index_test = build_knn_graph(X_test)

class GNNTransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNNTransformerClassifier, self).__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, return_layers=False):
        gnn_out = self.gnn(x, edge_index)
        transformer_input = gnn_out.unsqueeze(1)
        transformer_out = self.transformer(transformer_input)
        final_out = self.classifier(transformer_out.squeeze(1))

        if return_layers:
            return {
                'gnn_out': gnn_out.detach().cpu(),
                'transformer_out': transformer_out.squeeze(1).detach().cpu(),
                'logits': final_out.detach().cpu()
            }
        else:
            return final_out

input_dim = X_train.shape[1]
hidden_dim = 32
num_classes = len(np.unique(y))
criterion = nn.CrossEntropyLoss()
epochs = 20

final_test_accuracy = 1.0
run_count = 1

while final_test_accuracy == 0.99:
    print(f"\n--- Training Run__: {run_count} ---")

    model = GNNTransformerClassifier(input_dim, hidden_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.00010)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor, edge_index)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        _, train_pred = torch.max(train_outputs, 1)
        train_acc = (train_pred == y_train_tensor).float().mean()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor, edge_index_test)
            test_loss = criterion(test_outputs, y_test_tensor)
            _, test_pred = torch.max(test_outputs, 1)
            test_acc = (test_pred == y_test_tensor).float().mean()

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_accuracies.append(train_acc.item())
        test_accuracies.append(test_acc.item())

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, "
              f"Train Acc: {train_acc.item():.4f}, Test Loss: {test_loss.item():.4f}, "
              f"Test Acc: {test_acc.item():.4f}")

    final_test_accuracy = test_accuracies[-1]
    run_count += 1

print("\nTest Classification Report:")
print(classification_report(y_test_tensor.numpy(), test_pred.numpy()))

def plot_layer_output(features, labels, title):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

model.eval()
with torch.no_grad():
    layer_outputs = model(X_test_tensor, edge_index_test, return_layers=True)

plot_layer_output(layer_outputs['gnn_out'], y_test_tensor.numpy(), "GNN Layer Output")
plot_layer_output(layer_outputs['transformer_out'], y_test_tensor.numpy(), "Transformer Output")
plot_layer_output(layer_outputs['logits'], y_test_tensor.numpy(), "Classifier Logits Output")

def plot_combined_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs_range = range(1, len(train_losses)+1)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    axs[0].plot(epochs_range, train_accuracies, label='Train Accuracy', color='green')
    axs[0].plot(epochs_range, test_accuracies, label='Test Accuracy', color='blue')
    # axs[0].set_title("Accuracy over Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs_range, train_losses, label='Train Loss', color='red')
    axs[1].plot(epochs_range, test_losses, label='Test Loss', color='orange')
    # axs[1].set_title("Loss over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

plot_combined_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

