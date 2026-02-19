import h5py
import numpy as np
import torch
import scipy.io as sio
from scipy.signal import find_peaks
from DeepMathedFilterModel import DeepMatchedDetector

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



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tolerance = 30
#set r peak sensitivity parameter
R_peak_threshold = 0.25
min_peak_distance = 30


result = []
N = len(X_list)
for i in range(N):
    print(i)
    metrics = {
        "tp_deepMF": 0,
        "fp_deepMF": 0,
        "fn_deepMF": 0,
        "tp_deepMF_He": 0,
        "fp_deepMF_He": 0,
        "fn_deepMF_He": 0,
        "tp_rand": 0,
        "fp_rand": 0,
        "fn_rand": 0
    }
    deep_Matched_Model = DeepMatchedDetector()
    deep_Matched_Model_File = "./ERPDetectors/DeepMatchedFilterERPDetector_" + str(i) + ".pth"
    deep_Matched_Model.load_state_dict(torch.load(deep_Matched_Model_File, map_location=device))

    deep_Matched_Model_He = DeepMatchedDetector()
    deep_Matched_Model_He_File = "./ERPDetectors/DeepMatchedFilterERPDetector_" + str(i) + "_He.pth"
    deep_Matched_Model_He.load_state_dict(torch.load(deep_Matched_Model_He_File, map_location=device))

    random_initialized_Model = DeepMatchedDetector()
    random_initialized_file = "./ERPDetectors/RandomERPDetector_" + str(i) + ".pth"
    random_initialized_Model.load_state_dict(torch.load(random_initialized_file, map_location=device))
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

        # Move to device
        x = x.to(device)
        random_initialized_Model.to(device)
        deep_Matched_Model.to(device)
        deep_Matched_Model_He.to(device)
        # Change shape from (500, 4) → (1, 4, 500)
        x = x.unsqueeze(0)

        # Disable gradients for inference
        with torch.no_grad():
            output_deep_MF = deep_Matched_Model(x)
            output_deep_MF = output_deep_MF.detach().cpu().numpy().squeeze()

            output_deep_MF_He = deep_Matched_Model_He(x)
            output_deep_MF_He = output_deep_MF_He.detach().cpu().numpy().squeeze()

            output_rand = random_initialized_Model(x)
            output_rand = output_rand.detach().cpu().numpy().squeeze()

        detected_peaks_deep_MF, peaks_amplitude = find_peaks(output_deep_MF, height=R_peak_threshold, distance=min_peak_distance)
        tp, fp, fn = compute_metrics(detected_peaks_deep_MF, target_peaks, tolerance=tolerance)
        metrics["tp_deepMF"] += tp
        metrics["fp_deepMF"] += fp
        metrics["fn_deepMF"] += fn

        detected_peaks_deep_MF_He, peaks_amplitude = find_peaks(output_deep_MF_He, height=R_peak_threshold, distance=min_peak_distance)
        tp, fp, fn = compute_metrics(detected_peaks_deep_MF_He, target_peaks, tolerance=tolerance)
        metrics["tp_deepMF_He"] += tp
        metrics["fp_deepMF_He"] += fp
        metrics["fn_deepMF_He"] += fn

        detected_peaks_rand, peaks_amplitude = find_peaks(output_rand, height=R_peak_threshold, distance=min_peak_distance)
        tp, fp, fn = compute_metrics(detected_peaks_rand, target_peaks, tolerance=tolerance)
        metrics["tp_rand"] += tp
        metrics["fp_rand"] += fp
        metrics["fn_rand"] += fn

    result.append(metrics)

# convert to numpy array (important!)
result_array = np.array(result)

# save to matlab file
sio.savemat('LOO_results.mat', {'results': result_array})