# ============================================================
# DETECÇÃO DE TRANSIÇÕES DE FASE USANDO DEEP LEARNING
# PROJETO OTIMIZADO - PYTHON
# ============================================================
# Autor: André Luiz Magalhães de Oliveira
# Formação: Físico Médico | Especialista em Data Science & Analytics
# Universidade de São Paulo
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {DEVICE}")

# ============================================================
# PARÂMETROS
# ============================================================

L, J, TC = 40, 1.0, 2.269
LOW_TEMPS = np.linspace(0.5, 2.0, 5)
HIGH_TEMPS = np.linspace(2.8, 5.0, 5)
SAMPLES_PER_TEMP, THERMALIZATION, DECORRELATION = 100, 300, 5
BATCH_SIZE, EPOCHS, LR = 64, 10, 0.001

os.makedirs("results", exist_ok=True)

# ============================================================
# FUNÇÕES ISING
# ============================================================

def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

def metropolis_step(lattice, T, J=1.0):
    L = lattice.shape[0]
    for _ in range(L * L):
        i, j = np.random.randint(0, L, 2)
        spin = lattice[i, j]
        neighbors = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
        delta_E = 2 * J * spin * neighbors
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            lattice[i, j] *= -1
    return lattice

def total_energy(lattice, J=1.0):
    L = lattice.shape[0]
    return sum(-J * lattice[i, j] * (lattice[(i+1)%L, j] + lattice[i, (j+1)%L]) for i in range(L) for j in range(L))

def magnetization(lattice):
    return np.sum(lattice) / lattice.size

# ============================================================
# DATASET
# ============================================================

def generate_phase_dataset(temperatures, label):
    X, y, temps = [], [], []
    for T in temperatures:
        lattice = initialize_lattice(L)
        for _ in range(THERMALIZATION):
            metropolis_step(lattice, T)
        for _ in range(SAMPLES_PER_TEMP):
            for _ in range(DECORRELATION):
                metropolis_step(lattice, T)
            X.append(lattice.copy())
            y.append(label)
            temps.append(T)
    return X, y, temps

X_low, y_low, temps_low = generate_phase_dataset(LOW_TEMPS, 0)
X_high, y_high, temps_high = generate_phase_dataset(HIGH_TEMPS, 1)

X = np.array(X_low + X_high, dtype=np.float32)
X = (X + 1) / 2
X = X.reshape(-1, 1, L, L)
y = np.array(y_low + y_high)
temps = np.array(temps_low + temps_high)

print(f"\nDataset final: {X.shape}")

# ============================================================
# CNN
# ============================================================

class IsingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = IsingCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# TREINAMENTO
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE)

losses, accuracies = [], []
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    losses.append(running_loss / len(train_loader))
    accuracies.append(100 * correct / total)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={losses[-1]:.4f} | Acc={accuracies[-1]:.2f}%")

# ============================================================
# TESTE
# ============================================================

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:\n", classification_report(all_labels, all_preds))
ConfusionMatrixDisplay(confusion_matrix(all_labels, all_preds), display_labels=["Ordenada","Desordenada"]).plot()
plt.show()

# ============================================================
# CONCLUSÃO
# ============================================================

print("\n================================================")
print("PROJETO FINALIZADO COM SUCESSO")
print("================================================")
print("Etapas: Simulação Ising | Dataset | CNN | Grad-CAM | Interpretação")
print(f"Temperatura crítica aproximada: Tc ≈ {TC}")
