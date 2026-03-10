import torch
import gnn


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred = model(data)
        loss = F.mse_loss(pred, data.y.view(-1))

        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs

    return loss_all / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    mse_all = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        mse = F.mse_loss(pred, data.y.view(-1), reduction='sum')
        mse_all += mse.item()

    rmse = (mse_all / len(loader.dataset)) ** 0.5
    return rmse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

best_val = float('inf')

for epoch in range(1, 101):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_rmse = eval_epoch(model, val_loader, device)

    print(f'Epoch {epoch:03d} | Train MSE {train_loss:.4f} | Val RMSE {val_rmse:.4f}')

    if val_rmse < best_val:
        best_val = val_rmse
        torch.save(model.state_dict(), 'best_gnn.pt')
