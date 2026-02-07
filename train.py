import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np
import copy
from functools import partial


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

        # Skip connection: concat original 5 inputs + 3 frozen outputs = 8
        # Wider trainable head, no final Sigmoid (using BCEWithLogitsLoss)
        self.added_layers = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            intermediate = self.frozen_net(x)
        # Skip connection: concatenate original input with frozen net output
        combined = torch.cat([x, intermediate], dim=1)
        return self.added_layers(combined)


# --- 3. Data Preprocessing (train + test) ---
train_df = pd.read_csv('i239e_project_train.csv')
test_df = pd.read_csv('i239e_project_test.csv')

# Separate features and labels
drop_cols_train = ['Name', 'Feature#9', 'Feature#7', 'Feature#1', 'Survived']
drop_cols_test = ['Name', 'Feature#9', 'Feature#7', 'Feature#1']
X_train_raw = train_df.drop(columns=drop_cols_train)
y_train = train_df['Survived'].values
X_test_raw = test_df.drop(columns=drop_cols_test)

# Impute missing values (fit on train, transform both)
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns) # Use transform here to apply same imputation as train

# Feature selection — top 5 by Mutual Information (fit on train)
mi_scorer = partial(mutual_info_classif, random_state=42)
selector = SelectKBest(score_func=mi_scorer, k=5)
X_train_sel = selector.fit_transform(X_train_imp, y_train)
X_test_sel = selector.transform(X_test_imp) # Use transform here to apply same selection as train
selected_features = X_train_raw.columns[selector.get_support()].tolist()
selected_features_test = X_test_raw.columns[selector.get_support()].tolist()
print(f"Selected 5 features: {selected_features}")
print(f"Selected 5 features (test): {selected_features_test}")

# Standard scaling (fit on train, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel) # Use transform here to apply same scaling as train

print(f"Train shape: {X_train_scaled.shape}")
print(f"Test shape:  {X_test_scaled.shape}")

# Train / Validation split (80/20, stratified)
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
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
# Weighted loss: pos_weight = num_negatives / num_positives = 0.616 / 0.384 ≈ 1.608
pos_weight = torch.tensor([1.608])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.added_layers.parameters(), lr=0.001, weight_decay=1e-4)

print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Frozen parameters:    {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")


# --- 6. Training Loop with Early Stopping + LR Scheduler ---
num_epochs = 200
patience = 20
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

best_val_loss = float('inf')
best_model_state = None
epochs_no_improve = 0

print(f"\n--- Training for up to {num_epochs} epochs (early stopping patience={patience}) ---")

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

    # Compute validation loss
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_loss = criterion(val_logits, y_val_tensor).item()

    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1:3d}/{num_epochs}]  Train Loss: {avg_loss:.4f}  Val Loss: {val_loss:.4f}  LR: {lr:.6f}")

    if epochs_no_improve >= patience:
        print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"  Restored best model (val_loss={best_val_loss:.4f})")


# --- 7. Evaluation on Validation Set with Threshold Tuning ---
model.eval()
with torch.no_grad():
    val_logits = model(X_val_tensor)
    y_probs = torch.sigmoid(val_logits)

# Sweep thresholds to find optimal F1
best_f1 = 0.0
best_threshold = 0.5
for t in np.arange(0.30, 0.71, 0.01):
    preds = (y_probs.numpy() > t).astype(float)
    f1 = f1_score(y_val_tensor.numpy(), preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n--- Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f}) ---")

y_pred = (y_probs > best_threshold).float()

print("\n--- Validation Results ---")
print(f"Accuracy: {accuracy_score(y_val_tensor.numpy(), y_pred.numpy()):.4f}\n")
print("Classification Report:")
print(classification_report(
    y_val_tensor.numpy(), y_pred.numpy(),
    target_names=['Not Survived', 'Survived']
))


# --- 8. Predict on Test Set (i239e_project_test.csv) ---
with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_probs = torch.sigmoid(test_logits)
    test_pred = (test_probs > best_threshold).int().squeeze().numpy()

# Save predictions
output = test_df[['Feature#1', 'Name']].copy()
output['Survived'] = test_pred
output.to_csv('test_predictions.csv', index=False)
print(f"\nTest predictions saved to test_predictions.csv ({len(test_pred)} samples)")
print(f"Predicted survival rate: {test_pred.mean():.2%}")
