import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from DeepMathedFilterModel import EncoderDecoder, DeepMatchedDetector

use_templates = False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Load pretrained model
pretrained_model = EncoderDecoder()
if use_templates:
    pretrained_model.load_state_dict(torch.load("TemplateInitializedEncoderDecoder.pth", map_location=device))
else:
    pretrained_model.load_state_dict(torch.load("RandomInitializedEncoderDecoder.pth", map_location=device))


dataFile = 'ERP_Detector_Dataset.mat'
with h5py.File(dataFile, 'r') as f:
    cell_refs = f['eeg_segments'][:]   # (26,1)

    X_list = []
    Y_list = []

    for i in range(cell_refs.shape[0]):

        struct_ref = cell_refs[i, 0]       # get reference
        struct = f[struct_ref]             # dereference

        X = f[struct['X'][()]] if isinstance(struct['X'][()], h5py.Reference) else struct['X'][()]
        Y = f[struct['Y'][()]] if isinstance(struct['Y'][()], h5py.Reference) else struct['Y'][()]
        X = np.transpose(X, (2, 1, 0)) 
        Y = np.transpose(Y, (2, 1, 0)) 
        X_list.append(np.array(X))
        Y_list.append(np.array(Y))

N = len(X_list)

for i in range(N):
     # ---- LOO split ----
    X_test = torch.tensor(X_list[i], dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_list[i], dtype=torch.float32).to(device)

    X_train = torch.tensor(np.concatenate([X_list[j] for j in range(N) if j != i], axis=0), dtype=torch.float32).to(device)
    Y_train = torch.tensor(np.concatenate([Y_list[j] for j in range(N) if j != i], axis=0), dtype=torch.float32).to(device)

    # ---- Datasets ----
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset  = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #Initialize detector using Encoder
    detector_model = DeepMatchedDetector().to(device)

    detector_model.encoder.load_state_dict(
        pretrained_model.encoder.state_dict()
    )
    
    criterion = torch.nn.BCELoss() 
    optimizer = torch.optim.Adam(detector_model.parameters(), lr=1e-3)

    for epoch in range(250):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)  # (batch, 4, 500)
            yb = yb.squeeze(1) 
            yb = yb.to(device)  # (batch,  500)

            output = detector_model(xb)  # (batch, 1, 500)

            loss = criterion(output, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        # Save model after training
    if use_templates:
        save_file = "DeepMatchedFilterERPDetector_" 
    else:
        save_file = "RandomERPDetector_"

    save_file = save_file + str(i) + ".pth"
    torch.save(detector_model.state_dict(), save_file)
    print("Model saved successfully.")