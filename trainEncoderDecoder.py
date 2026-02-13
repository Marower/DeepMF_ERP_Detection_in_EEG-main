import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from DeepMathedFilterModel import EncoderDecoder



print(torch.backends.mps.is_available())
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EncoderDecoder().to(device)
useTemplates = False
if useTemplates:
    templatesFile = 'channels_templates.mat'
    with h5py.File(templatesFile, 'r') as f:
        templates = f['templates'][:]   # change key if needed

    templates = np.array(templates)
    templates = templates.T
    print(templates.shape)
    templates = torch.tensor(templates, dtype=torch.float32)

    # assign
    with torch.no_grad():
        model.encoder[0].weight.copy_(templates)
        model.encoder[0].bias.zero_()


dataFile = 'eeg_dataset.mat'
with h5py.File(dataFile, 'r') as f:
    data = f['eeg_segments'][:]

#(samples, channels, time)
data = np.transpose(data, (2, 1, 0))
print(data.shape)

X = torch.tensor(data, dtype=torch.float32)
X = X.to(device)
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(250):
    for xb in loader:
        xb = xb[0].to(device)

        output = model(xb)
        loss = criterion(output, xb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model after training
if useTemplates:
    torch.save(model.state_dict(), "TemplateInitializedEncoderDecoder.pth")
    print("Model saved successfully.")
else:
    torch.save(model.state_dict(), "RandomInitializedEncoderDecoder.pth")
    print("Random model saved successfully.")

