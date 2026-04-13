import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from MLP import *
from texture_encoders import *
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader


def train_model(
    model: nn.Module,
    trainloader: DataLoader,
    lr: float = 1e-4,
    epochs: int = 100
) -> nn.Module:
    """
    Train a PyTorch model using a classification objective.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    trainloader : DataLoader
        PyTorch DataLoader providing (inputs, labels) batches.
    lr : float, optional
        Learning rate for the optimizer, by default 1e-4.
    epochs : int, optional
        Number of training epochs, by default 100.

    Returns
    -------
    nn.Module
        The trained model.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_losses = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in trainloader:
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model.forward(x_batch)

            # Compute loss
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Store cumulative loss
            total_losses.append(total_loss)

    # Plot training loss evolution
    plt.figure()
    plt.plot(total_losses)
    plt.title('Total losses during training')
    plt.show()

    return model


def test_model(
    model: nn.Module,
    testloader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a PyTorch model on a test dataset.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    testloader : DataLoader
        DataLoader providing test batches.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - true labels
        - predicted logits (or probabilities depending on model output)
    """
    trues = []
    preds = []

    # Iterate over test batches
    for x_batch, y_batch in testloader:
        batch_preds = model(x_batch)

        # Move tensors to CPU and convert to numpy
        preds.append(batch_preds.cpu().detach().numpy())
        trues.append(y_batch.cpu().detach().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return trues, preds


def train_test_split(
    output_path: str,
    id_path: str,
    split: int = 0,
    lidar: bool = True,
    aux_data_source_pths: list[str] = [],
    models: list[str] = ['RF']
) -> dict:
    """
    Train and evaluate multiple models (RF, XGBoost, MLP) on a given data split.

    This function:
    - Loads precomputed train/val/test splits
    - Optionally merges auxiliary data sources
    - Trains selected models
    - Evaluates them using classification metrics

    Parameters
    ----------
    output_path : str
        Path to directory containing split data.
    id_path : str
        Path to IDs (currently unused in function).
    split : int, optional
        Index of the data split, by default 0.
    lidar : bool, optional
        Whether LiDAR data is included (affects join strategy), by default True.
    aux_data_source_pths : list[str], optional
        List of paths to auxiliary datasets, by default [].
    models : list[str], optional
        Models to train ('RF', 'XGBoost', 'MLP'), by default ['RF'].

    Returns
    -------
    dict
        Dictionary mapping model names to classification report arrays.
    """
    cr_dict = {}

    # Load datasets
    df_train = pd.read_pickle(
        f"{output_path}/split_{split+1}/train_texture_features.pkl"
    ).set_index('id')

    df_val = pd.read_pickle(
        f"{output_path}/split_{split+1}/val_texture_features.pkl"
    ).set_index('id')

    df_test = pd.read_pickle(
        f"{output_path}/split_{split+1}/test_texture_features.pkl"
    ).set_index('id')

    # Merge auxiliary data sources
    for i, pth in enumerate(aux_data_source_pths):

        # Normalize data
        norm_vals_1, norm_vals_2 = get_norm_vals([pth])
        ids = np.load(f'{pth}/ids.npy', allow_pickle=True).astype('int')
        data = normalize(np.load(pth), norm_vals_1, norm_vals_2)

        # Flatten spatial dimensions
        data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

        # Convert to DataFrame indexed by IDs
        data = pd.DataFrame(data, index=ids)
        data.index.name = 'id'

        # Join strategy depends on lidar flag
        if not lidar and i == 0:
            df_train.join(data, how='right', rsuffix=f'_{i}').dropna(inplace=True)
            df_val.join(data, how='right', rsuffix=f'_{i}').dropna(inplace=True)
            df_test.join(data, how='right', rsuffix=f'_{i}').dropna(inplace=True)
        else:
            df_train.join(data, how='inner', rsuffix=f'_{i}').dropna(inplace=True)
            df_val.join(data, how='inner', rsuffix=f'_{i}').dropna(inplace=True)
            df_test.join(data, how='inner', rsuffix=f'_{i}').dropna(inplace=True)

    print('===================')
    print(f'Split {split}')
    print('===================')

    # ------------------
    # Random Forest
    # ------------------
    if 'RF' in models:
        print('Random Forest')

        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8)

        rf_clf.fit(
            df_train.drop(columns=['label']).to_numpy(),
            df_train['label'].to_numpy()
        )

        preds = rf_clf.predict(
            df_test.drop(columns=['label']).to_numpy()
        )

        cr = pd.DataFrame(
            classification_report(preds, df_test['label'].to_numpy(), output_dict=True)
        ).round(2).to_numpy()

        cr_dict['RF'] = cr

        print(confusion_matrix(preds, df_test['label'].to_numpy()))
        print(f'{cr}\n')

    # ------------------
    # Gradient Boosting (XGBoost-like)
    # ------------------
    if 'XGBoost' in models:
        print('XGBoost')

        XGB_clf = GradientBoostingClassifier()

        XGB_clf.fit(
            df_train.drop(columns=['label']).to_numpy(),
            df_train['label'].to_numpy()
        )

        preds = XGB_clf.predict(
            df_test.drop(columns=['label']).to_numpy()
        )

        cr = pd.DataFrame(
            classification_report(preds, df_test['label'].to_numpy(), output_dict=True)
        ).round(2).to_numpy()

        cr_dict['XGBoost'] = cr

        print(confusion_matrix(preds, df_test['label'].to_numpy()))
        print(f'{cr}\n')

    # ------------------
    # MLP (PyTorch)
    # ------------------
    if 'MLP' in models:
        print('MLP')

        # Prepare training data
        try:
            x_train = df_train.drop(columns=["label", "id"]).values
        except:
            x_train = df_train.drop(columns=["label"]).values

        y_train = df_train["label"].values

        train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Prepare validation data
        try:
            x_val = df_val.drop(columns=["label", "id"]).values
        except:
            x_val = df_val.drop(columns=["label"]).values

        y_val = df_val["label"].values

        val_dataset = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        # Train MLP model (external function)
        MLP_model = MLP()
        MLP_model = train_mlp(train_loader, val_loader)

        # Prepare test data
        try:
            x_test = df_test.drop(columns=["label", "id"]).values
        except:
            x_test = df_test.drop(columns=["label"]).values

        y_test = df_test["label"].values

        test_dataset = TensorDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluate MLP
        trues, preds = evaluate_mlp(MLP_model, test_loader)

        cr = pd.DataFrame(
            classification_report(preds, trues, output_dict=True)
        ).round(2).to_numpy()

        cr_dict['MLP'] = cr

        print(confusion_matrix(preds, trues))
        print(f'{cr}\n')

    return cr_dict