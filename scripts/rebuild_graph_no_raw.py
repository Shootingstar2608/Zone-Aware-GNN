import os
import torch
import numpy as np
import json

def rebuild_dataset(t_out_new, save_path):
    # Load original dataset
    d = torch.load('data/processed/graph_dataset.pt', weights_only=False)
    X = d['X'].numpy() # (S, N, T_in*F)
    Y = d['Y'].numpy() # (S, N, T_out)
    TL = d['time_labels'].numpy() # (S,)
    
    S_orig = X.shape[0]
    N = X.shape[1]
    T_in = 12
    F = 4
    T_out_orig = Y.shape[2]
    
    # 1. Reconstruct full sequence of time_labels (TL_full)
    # TL_full should be of length T = S_orig + T_in + T_out_orig - 1
    # Actually, TL is the time label of the LAST step of the input window.
    # build_graph.py:
    # idx = t + t_in - 1
    # T_w.append(times[idx])
    # So TL[s] corresponds to time t = s + T_in - 1
    # The length of times was T.
    # T = S_orig + T_in + T_out_orig - 1 = 658 + 12 + 3 - 1 = 672
    T = S_orig + T_in + T_out_orig - 1
    
    # Let's reconstruct X_full for all available steps
    # X_full will have shape (T, N, F). We can only reconstruct up to T-T_out_orig for all features.
    X_full = np.zeros((T, N, F))
    
    # Fill X_full from X
    for s in range(S_orig):
        # X[s] is (N, T_in*F) -> reshape to (N, T_in, F) -> transpose to (T_in, N, F)
        x_window = X[s].reshape(N, T_in, F).transpose(1, 0, 2)
        # It corresponds to X_full[s : s+T_in]
        # We can just use it to fill
        X_full[s : s+T_in] = x_window
        
    # Y gives us feature 0 for the remaining T_out_orig steps
    # Y[s] is (N, T_out_orig) -> Y_window is X_full[s+T_in : s+T_in+T_out_orig, :, 0]
    for s in range(S_orig):
        y_window = Y[s].T # (T_out_orig, N)
        X_full[s+T_in : s+T_in+T_out_orig, :, 0] = y_window
        
    # What about TL_full?
    TL_full = np.zeros(T, dtype=np.int64)
    # TL[s] = TL_full[s + T_in - 1]
    for s in range(S_orig):
        TL_full[s + T_in - 1] = TL[s]
        
    # We might not know TL_full for the last few steps, but we don't need them if we increase T_out
    # Because new S will be smaller: S_new = T - T_in - t_out_new + 1
    # If t_out_new > T_out_orig, S_new < S_orig.
    # The required indices for TL in the new dataset will be:
    # s_new + T_in - 1. The max index is S_new - 1 + T_in - 1 = T - t_out_new - 1.
    # Since t_out_new >= T_out_orig (6 >= 3), T - t_out_new - 1 <= T - 3 - 1.
    # We already know TL up to index S_orig - 1 + T_in - 1 = T - 3 - 1.
    # So we have ALL required TL values!
    
    # 2. Reconstruct hour_list, dow_list, time_sinc_list if they exist
    # Wait, the dataset was generated WITHOUT hour, dow, time_sinc because the command failed!
    # Let me check if they exist in `d`.
    # 'hour', 'dow', 'time_sinc' were NOT in dict_keys(['A', 'Z', 'X', 'Y', 'time_labels', 'nodes', 'feature_names', 'zone_types'])
    # That means the previous dataset was from the OLD code, before Bao added them!
    
    # Wait! If we don't have hour and dow, we CANNOT run `zone_full_sinc` properly because it needs them!
    # No, wait, in test_bao.py, it says Sinusoidal time encoder takes `time_idx` (which is `TL`!)
    # Let's check `models/time_zone_aware_gnn.py` again.
    
    # 3. Create new samples
    S_new = T - T_in - t_out_new + 1
    X_w, Y_w, T_w = [], [], []
    
    for t_step in range(S_new):
        x_window = X_full[t_step : t_step + T_in]
        x_flat = x_window.transpose(1, 0, 2).reshape(N, -1)
        
        y_window = X_full[t_step + T_in : t_step + T_in + t_out_new, :, 0]
        y_flat = y_window.T
        
        idx = t_step + T_in - 1
        
        X_w.append(x_flat)
        Y_w.append(y_flat)
        T_w.append(TL_full[idx])
        
    d_new = {
        'A': d['A'],
        'Z': d['Z'],
        'X': torch.tensor(np.stack(X_w), dtype=torch.float32),
        'Y': torch.tensor(np.stack(Y_w), dtype=torch.float32),
        'time_labels': torch.tensor(np.array(T_w), dtype=torch.long),
        'nodes': d['nodes'],
        'feature_names': d['feature_names'],
        'zone_types': d['zone_types']
    }
    
    torch.save(d_new, save_path)
    
    # Update meta.json
    with open('data/processed/meta.json') as f:
        meta = json.load(f)
    meta['T_out'] = t_out_new
    meta['S'] = S_new
    with open('data/processed/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"Rebuilt dataset for T_out={t_out_new}: X shape {d_new['X'].shape}, Y shape {d_new['Y'].shape}")

if __name__ == "__main__":
    import sys
    t_out = int(sys.argv[1])
    rebuild_dataset(t_out, 'data/processed/graph_dataset.pt')
