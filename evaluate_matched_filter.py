import h5py
import numpy as np
import torch
import scipy.io as sio
from scipy.signal import find_peaks
from TraditionalMatchedFilter import TraditionalMatchedFilter

def compute_metrics(detected_peaks, target_peaks, tolerance=20):
    detected_peaks = sorted(detected_peaks)
    target_peaks = sorted(target_peaks)

    tp = 0
    used_detected = set()

    det_idx = 0

    for target in target_peaks:
        # Move detection index forward if detections are too early
        while det_idx < len(detected_peaks) and detected_peaks[det_idx] < target - tolerance:
            det_idx += 1

        # Try to match first valid detection in window
        match_found = False
        check_idx = det_idx

        while check_idx < len(detected_peaks) and detected_peaks[check_idx] <= target + tolerance:
            if check_idx not in used_detected:
                tp += 1
                used_detected.add(check_idx)
                match_found = True
                break
            check_idx += 1

        # If no match found → FN
        # (we'll compute FN after loop)

    fp = len(detected_peaks) - len(used_detected)
    fn = len(target_peaks) - tp

    return tp, fp, fn

templateFile = 'channels_templates_HE.mat'
# Load the file
# Open the v7.3 file using h5py
with h5py.File(templateFile, 'r') as f:
    # Option A: Load a specific variable by name
    # Note: h5py loads arrays transposed compared to scipy/MATLAB
    templates = np.array(f['templates']).T


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

tolerance = 30
#set r peak sensitivity parameter
R_peak_threshold = 0.25
min_peak_distance = 30


result = []
N = len(X_list)
classical_MF = TraditionalMatchedFilter(templates=templates)
classical_MF.eval()
for i in range(N):
    print(i)
    metrics = {
        "tp_MF": 0,
        "fp_MF": 0,
        "fn_MF": 0,
    }

    Y = Y_list[i]
    X = X_list[i]

    for idx in range(X.shape[0]):
        target = Y[idx,0,:]

        target_peaks, peaks_amplitude = find_peaks(target, height=R_peak_threshold, distance=min_peak_distance)

        x = X[idx,:,:]
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()


        # Change shape from (500, 4) → (1, 4, 500)
        x = x.unsqueeze(0)

        # Disable gradients for inference
        with torch.no_grad():
            output_MF = classical_MF(x)
            output_MF = output_MF.detach().cpu().numpy().squeeze()


        detected_peaks_deep_MF, peaks_amplitude = find_peaks(output_MF, height=R_peak_threshold, distance=min_peak_distance)
        tp, fp, fn = compute_metrics(detected_peaks_deep_MF, target_peaks, tolerance=tolerance)
        metrics["tp_MF"] += tp
        metrics["fp_MF"] += fp
        metrics["fn_MF"] += fn

    result.append(metrics)

# convert to numpy array (important!)
result_array = np.array(result)

# save to matlab file
sio.savemat('matched_filter_results.mat', {'results': result_array})