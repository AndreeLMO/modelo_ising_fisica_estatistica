# ============================================================
# PROJETO COMPLETO
# MODELO DE ISING 2D + CNN + GRAD-CAM
# ============================================================
# Projeto:
# Detecção de Transições de Fase usando Deep Learning
#
# Etapas:
# 1. Simulação do Modelo de Ising 2D
# 2. Geração do Dataset
# 3. Deep Learning com CNN
# 4. Grad-CAM para interpretabilidade
#
# Otimizações:
# - Dataset reduzido
# - Uso de float32/int8
# - torch.no_grad()
# - Sem salvar imagens intermediárias
# - Estrutura modular
#
# Autor: André Luiz
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDispositivo: {DEVICE}")

# ============================================================
# PARÂMETROS FÍSICOS
# ============================================================

L = 40
J = 1.0

TC = 2.269

# ============================================================
# PARÂMETROS DATASET
# ============================================================

LOW_TEMPS = np.linspace(0.5, 2.0, 5)

HIGH_TEMPS = np.linspace(2.8, 5.0, 5)

SAMPLES_PER_TEMP = 100

THERMALIZATION = 300
DECORRELATION = 5

# ============================================================
# PARÂMETROS CNN
# ============================================================

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# ============================================================
# PASTAS
# ============================================================

os.makedirs("results", exist_ok=True)

# ============================================================
# FUNÇÕES DO MODELO DE ISING
# ============================================================

def initialize_lattice(L):

    return np.random.choice(
        [-1, 1],
        size=(L, L)
    ).astype(np.int8)


def metropolis_step(lattice, T, J=1.0):

    L = lattice.shape[0]

    for _ in range(L * L):

        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        spin = lattice[i, j]

        neighbors = (
            lattice[(i + 1) % L, j] +
            lattice[(i - 1) % L, j] +
            lattice[i, (j + 1) % L] +
            lattice[i, (j - 1) % L]
        )

        delta_E = 2 * J * spin * neighbors

        if delta_E < 0:

            lattice[i, j] *= -1

        elif np.random.rand() < np.exp(-delta_E / T):

            lattice[i, j] *= -1

    return lattice


def total_energy(lattice, J=1.0):

    L = lattice.shape[0]

    energy = 0

    for i in range(L):

        for j in range(L):

            spin = lattice[i, j]

            neighbors = (
                lattice[(i + 1) % L, j] +
                lattice[i, (j + 1) % L]
            )

            energy += -J * spin * neighbors

    return energy


def magnetization(lattice):

    return np.sum(lattice) / lattice.size


# ============================================================
# ETAPA 1 — SIMULAÇÃO ISING
# ============================================================

print("\n================================================")
print("ETAPA 1 — SIMULAÇÃO DO MODELO DE ISING")
print("================================================")

T_demo = 2.2

lattice = initialize_lattice(L)

energies = []
magnetizations = []

for step in tqdm(range(300)):

    metropolis_step(lattice, T_demo, J)

    E = total_energy(lattice, J)

    M = abs(magnetization(lattice))

    energies.append(E)
    magnetizations.append(M)

# ============================================================
# GRÁFICO LATTICE FINAL
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(lattice, cmap='coolwarm')

plt.title(f"Modelo de Ising 2D | T={T_demo}")

plt.axis('off')

plt.colorbar()

plt.show()

# ============================================================
# ENERGIA
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(energies)

plt.title("Energia do Sistema")

plt.xlabel("Passos Monte Carlo")

plt.ylabel("Energia")

plt.grid()

plt.show()

# ============================================================
# MAGNETIZAÇÃO
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(magnetizations)

plt.title("Magnetização")

plt.xlabel("Passos Monte Carlo")

plt.ylabel("Magnetização")

plt.grid()

plt.show()

# ============================================================
# ETAPA 2 — GERAÇÃO DATASET
# ============================================================

print("\n================================================")
print("ETAPA 2 — GERAÇÃO DO DATASET")
print("================================================")

X = []
y = []
temps = []

def generate_phase_dataset(temperatures, label):

    global X, y, temps

    for T in temperatures:

        print(f"\nTemperatura = {T:.2f}")

        lattice = initialize_lattice(L)

        # thermalization
        for _ in range(THERMALIZATION):

            metropolis_step(lattice, T)

        # coleta
        for _ in tqdm(range(SAMPLES_PER_TEMP)):

            for _ in range(DECORRELATION):

                metropolis_step(lattice, T)

            X.append(lattice.copy())

            y.append(label)

            temps.append(T)

# ordenado
generate_phase_dataset(LOW_TEMPS, 0)

# desordenado
generate_phase_dataset(HIGH_TEMPS, 1)

# ============================================================
# CONVERTER
# ============================================================

X = np.array(X).astype(np.float32)

X = (X + 1) / 2

X = X.reshape(-1, 1, L, L)

y = np.array(y)

temps = np.array(temps)

print(f"\nDataset final: {X.shape}")

# ============================================================
# VISUALIZAÇÃO DO DATASET
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

axes = axes.flatten()

for i in range(10):

    idx = np.random.randint(0, len(X))

    axes[i].imshow(X[idx][0], cmap='coolwarm')

    axes[i].set_title(
        f"T={temps[idx]:.2f}\nClasse={y[idx]}"
    )

    axes[i].axis('off')

plt.tight_layout()

plt.show()

# ============================================================
# DISTRIBUIÇÃO CLASSES
# ============================================================

plt.figure(figsize=(5, 4))

classes, counts = np.unique(y, return_counts=True)

plt.bar(classes, counts)

plt.xticks([0, 1], ["Ordenada", "Desordenada"])

plt.title("Distribuição das Classes")

plt.show()

# ============================================================
# ETAPA 3 — CNN
# ============================================================

print("\n================================================")
print("ETAPA 3 — CNN")
print("================================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)

# ============================================================
# CNN
# ============================================================

class IsingCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 5 * 5, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

model = IsingCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TREINAMENTO
# ============================================================

losses = []
accuracies = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = 100 * correct / total

    losses.append(epoch_loss)
    accuracies.append(epoch_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss={epoch_loss:.4f} | "
        f"Acc={epoch_acc:.2f}%"
    )

# ============================================================
# TESTE
# ============================================================

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.numpy())

# ============================================================
# MATRIZ CONFUSÃO
# ============================================================

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Ordenada",
        "Desordenada"
    ]
)

fig, ax = plt.subplots(figsize=(5, 5))

disp.plot(ax=ax)

plt.title("Matriz de Confusão")

plt.show()

# ============================================================
# RELATÓRIO
# ============================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(
    classification_report(
        all_labels,
        all_preds
    )
)

# ============================================================
# LOSS
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(losses)

plt.title("Loss de Treinamento")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid()

plt.show()

# ============================================================
# ACCURACY
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(accuracies)

plt.title("Accuracy de Treinamento")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid()

plt.show()

# ============================================================
# PREDIÇÕES
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

axes = axes.flatten()

indices = np.random.choice(len(X_test), 10)

with torch.no_grad():

    for i, idx in enumerate(indices):

        image = X_test[idx].unsqueeze(0).to(DEVICE)

        output = model(image)

        pred = torch.argmax(output, dim=1).item()

        axes[i].imshow(
            X_test[idx][0],
            cmap='coolwarm'
        )

        axes[i].set_title(
            f"Real={y_test[idx].item()}\nPred={pred}"
        )

        axes[i].axis('off')

plt.tight_layout()

plt.show()

# ============================================================
# ETAPA 4 — GRAD-CAM
# ============================================================

print("\n================================================")
print("ETAPA 4 — GRAD-CAM")
print("================================================")

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.forward_hook
        )

        target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, input, output):

        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, x, target_class=None):

        output = self.model(x)

        if target_class is None:

            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()

        loss = output[:, target_class]

        loss.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        ).to(DEVICE)

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        cam /= cam.max() + 1e-8

        return cam.detach().cpu().numpy()

target_layer = model.features[6]

gradcam = GradCAM(model, target_layer)

# ============================================================
# VISUALIZAÇÃO GRAD-CAM
# ============================================================

NUM_EXAMPLES = 4

indices = np.random.choice(
    len(X_test),
    NUM_EXAMPLES,
    replace=False
)

fig, axes = plt.subplots(
    NUM_EXAMPLES,
    3,
    figsize=(12, 4 * NUM_EXAMPLES)
)

for row, idx in enumerate(indices):

    image = X_test[idx].unsqueeze(0).to(DEVICE)

    output = model(image)

    pred = torch.argmax(output, dim=1).item()

    heatmap = gradcam.generate(
        image,
        target_class=pred
    )

    lattice = X_test[idx][0].cpu().numpy()

    # lattice
    axes[row, 0].imshow(
        lattice,
        cmap='coolwarm'
    )

    axes[row, 0].set_title(
        f"Lattice\nReal={y_test[idx].item()}"
    )

    axes[row, 0].axis('off')

    # heatmap
    axes[row, 1].imshow(
        heatmap,
        cmap='jet'
    )

    axes[row, 1].set_title("Grad-CAM")

    axes[row, 1].axis('off')

    # overlay
    axes[row, 2].imshow(
        lattice,
        cmap='gray'
    )

    axes[row, 2].imshow(
        heatmap,
        cmap='jet',
        alpha=0.5
    )

    axes[row, 2].set_title(
        f"Predição={pred}"
    )

    axes[row, 2].axis('off')

plt.tight_layout()

plt.show()

# ============================================================
# CONCLUSÃO
# ============================================================

print("\n================================================")
print("PROJETO FINALIZADO")
print("================================================")

print("""
Etapas implementadas:

1. Modelo de Ising 2D
2. Algoritmo de Metropolis
3. Energia e magnetização
4. Geração do dataset
5. CNN em PyTorch
6. Classificação de fases
7. Grad-CAM
8. IA interpretável

A CNN aprendeu padrões físicos relacionados
à transição de fase do Modelo de Ising.

Temperatura crítica aproximada:
Tc ≈ 2.269
""")