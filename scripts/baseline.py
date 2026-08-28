import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.data import get_data_loaders

train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

def loader_to_numpy(loader):
    X, y = [], []
    for images, labels in loader:
        # images: [B, 1, 28, 28] -> flatten each to 784
        X.append(images.view(images.size(0), -1).numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

X_train, y_train = loader_to_numpy(train_loader)
X_val,   y_val   = loader_to_numpy(val_loader)

clf = LogisticRegression(max_iter=1000)   # may need max_iter high — 784 features
clf.fit(X_train, y_train)
print("baseline val accuracy:", accuracy_score(y_val, clf.predict(X_val)))