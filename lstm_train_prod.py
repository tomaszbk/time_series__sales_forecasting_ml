from data_pipeline import data_transform_pipeline, cap_sales
from lstm_model import model
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np


data = pd.read_csv('data/train.csv')

X = data
y = X.pop('sales')

y_transformed = cap_sales(y, 7000)
X_transformed = data_transform_pipeline.fit_transform(X)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

X_tensor = torch.tensor(X_transformed.astype(np.float32)).to(device)
y_tensor = torch.from_numpy(y_transformed.values.astype(np.float32)).reshape(-1, 1).to(device)


batch_size = 5000
num_epochs = 10

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Entrenar el modelo
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in dataloader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


X_competition = pd.read_csv('data/test.csv')
competition_indexes = X_competition['id']
X_competition = data_transform_pipeline.transform(X_competition)

model.eval()
with torch.no_grad():
    competition_predictions = model(torch.tensor(X_competition.astype(np.float32)).to(device))

competition_predictions[competition_predictions < 0] = 0

competition_predictions = competition_predictions.cpu().numpy().reshape(-1)
pd.DataFrame({'id':competition_indexes, 
              'sales':competition_predictions }).to_csv('data/lstm_submission.csv', index=False)