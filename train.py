import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np


# --- 1. Define the Pretrained Network Architecture ---
# Required so torch.load can reconstruct the saved model object
class PretrainedNetwork(nn.Module):
    def __init__(self):
        super(PretrainedNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. Define the Full Model with Frozen + Trainable Layers ---
class TitanicModel(nn.Module):
    def __init__(self, frozen_net):
        super(TitanicModel, self).__init__()

        # Frozen Network: 5 inputs -> 3 outputs
        self.frozen_net = frozen_net
        for param in self.frozen_net.parameters():
            param.requires_grad = False

        # Added Trainable Layers: 3 inputs (from frozen net) -> 1 output
        self.added_layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            intermediate = self.frozen_net(x)
        return self.added_layers(intermediate)


# --- 3. Data Preprocessing (train + test) ---
train_df = pd.read_csv('i239e_project_train.csv')
test_df = pd.read_csv('i239e_project_test.csv')

# Separate features and labels
drop_cols_train = ['Name', 'Feature#9', 'Feature#7', 'Feature#1', 'Survived']
drop_cols_test = ['Name', 'Feature#9', 'Feature#7', 'Feature#1']
X_train_raw = train_df.drop(columns=drop_cols_train)
y = train_df['Survived'].values
X_test_raw = test_df.drop(columns=drop_cols_test)

# Impute missing values (fit on train, transform both)
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

# Feature selection — top 5 by Mutual Information (fit on train)
selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_train_sel = selector.fit_transform(X_train_imp, y)
X_test_sel = selector.transform(X_test_imp)
selected_features = X_train_raw.columns[selector.get_support()].tolist()
print(f"Selected 5 features: {selected_features}")

# Standard scaling (fit on train, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

print(f"Train shape: {X_train_scaled.shape}")
print(f"Test shape:  {X_test_scaled.shape}")

# Train / Validation split (80/20, stratified)
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_split)
y_train_tensor = torch.FloatTensor(y_train_split).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# DataLoader for mini-batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# --- 4. Load Frozen Pretrained Network ---
frozen_net = torch.load('pretrained_network.pth', map_location='cpu', weights_only=False)
frozen_net.eval()
print(f"\nLoaded frozen network:\n{frozen_net}")


# --- 5. Build Model ---
model = TitanicModel(frozen_net)

# Loss and Optimizer — only train the added layers (frozen net stays fixed)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.added_layers.parameters(), lr=0.001)

print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Frozen parameters:    {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")


# --- 6. Training Loop ---
num_epochs = 50
print(f"\n--- Training for {num_epochs} epochs ---")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch [{epoch+1:3d}/{num_epochs}]  Loss: {avg_loss:.4f}")


# --- 7. Evaluation on Validation Set ---
model.eval()
with torch.no_grad():
    y_probs = model(X_val_tensor)
    y_pred = (y_probs > 0.5).float()

print("\n--- Validation Results ---")
print(f"Accuracy: {accuracy_score(y_val_tensor.numpy(), y_pred.numpy()):.4f}\n")
print("Classification Report:")
print(classification_report(
    y_val_tensor.numpy(), y_pred.numpy(),
    target_names=['Not Survived', 'Survived']
))


# --- 8. Predict on Test Set (i239e_project_test.csv) ---
with torch.no_grad():
    test_probs = model(X_test_tensor)
    test_pred = (test_probs > 0.5).int().squeeze().numpy()

# Save predictions
output = test_df[['Feature#1', 'Name']].copy()
output['Survived'] = test_pred
output.to_csv('test_predictions.csv', index=False)
print(f"\nTest predictions saved to test_predictions.csv ({len(test_pred)} samples)")
print(f"Predicted survival rate: {test_pred.mean():.2%}")

