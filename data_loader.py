import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'processed_data')
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)


def load_dsads():
    print("\n[1/4] DSADS (Torso) -> Loading...")
    dsads_root = None
    paths = [
        os.path.join(DATA_ROOT, 'DailyandSportActivitiesDataset', 'OriginalData'),
        os.path.join(DATA_ROOT, 'data', 'DailyandSportActivitiesDataset', 'OriginalData')
    ]
    for p in paths:
        if os.path.exists(p): dsads_root = p; break
    if not dsads_root:
        for root, dirs, files in os.walk(DATA_ROOT):
            if 'a01' in dirs: dsads_root = root; break
    if not dsads_root: print("DSADS Not Found"); return

    # Torso
    target_map = {'a09': 0, 'a05': 1, 'a06': 2, 'a03': 3}
    all_data, all_labels = [], []

    for act, lbl in target_map.items():
        p = os.path.join(dsads_root, act)
        if not os.path.exists(p): continue
        for root, _, files in os.walk(p):
            for f in files:
                if f.endswith('.txt'):
                    try:
                        d = np.loadtxt(os.path.join(root, f), delimiter=',')
                        # Torso: Acc(0-3) + Gyro(3-6)
                        acc = d[:, 0:3]
                        gyro = d[:, 3:6]
                        d = np.hstack([acc, gyro])
                        all_data.append(d)
                        all_labels.append(np.full(len(d), lbl))
                    except:
                        pass
    if all_data:
        X, y = np.vstack(all_data), np.concatenate(all_labels)
        np.save(os.path.join(SAVE_DIR, 'target_features_dsads_raw.npy'), {'X': X, 'y': y})
        print(f"DSADS (Torso) Loaded. Shape: {X.shape}")


def load_pamap2():
    print("\n[2/4] PAMAP2 (Chest) -> Loading...")
    pamap_root = None
    hard_path = os.path.join(DATA_ROOT, 'PAMAP2Dataset', 'Protocol')
    if os.path.exists(hard_path):
        pamap_root = hard_path
    else:
        files = glob.glob(os.path.join(DATA_ROOT, '**', 'subject101.dat'), recursive=True)
        if files: pamap_root = os.path.dirname(files[0])
    if not pamap_root: print(" PAMAP2 Not Found"); return

    id_map = {4: 0, 12: 1, 13: 2, 1: 3}
    all_data, all_labels = [], []
    dat_files = glob.glob(os.path.join(pamap_root, '*.dat'))

    for f in dat_files:
        try:
            df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
            raw = df.values
            for pid, lbl in id_map.items():
                mask = (raw[:, 1] == pid)
                if np.sum(mask) > 0:
                    # Chest: Acc(21-24), Gyro(27-30)
                    acc = raw[mask, 21:24]
                    gyro = raw[mask, 27:30]
                    d = np.hstack([acc, gyro])
                    if np.isnan(d).all(): continue
                    d = d[~np.isnan(d).any(axis=1)]
                    d = d[::4]
                    all_data.append(d)
                    all_labels.append(np.full(len(d), lbl))
        except:
            pass
    if all_data:
        X, y = np.vstack(all_data), np.concatenate(all_labels)
        np.save(os.path.join(SAVE_DIR, 'source_features_pamap2_raw.npy'), {'X': X, 'y': y})
        print(f" PAMAP2 (Chest) Loaded. Shape: {X.shape}")


def load_uci():

    print("\n[3/4] UCI (Waist) -> Loading...")
    uci_root = None
    files = glob.glob(os.path.join(DATA_ROOT, '**', 'Inertial Signals', 'total_acc_x_train.txt'), recursive=True)
    if files: uci_root = os.path.dirname(os.path.dirname(os.path.dirname(files[0])))
    if not uci_root: print(" UCI Not Found"); return

    target_map = {1: 0, 2: 1, 3: 2, 6: 3}
    for split in ['train', 'test']:
        try:
            y_path = os.path.join(uci_root, split, f'y_{split}.txt')
            if not os.path.exists(y_path): continue
            y_raw = np.loadtxt(y_path)
            ax = np.loadtxt(os.path.join(uci_root, split, 'Inertial Signals', f'total_acc_x_{split}.txt'))
            ay = np.loadtxt(os.path.join(uci_root, split, 'Inertial Signals', f'total_acc_y_{split}.txt'))
            az = np.loadtxt(os.path.join(uci_root, split, 'Inertial Signals', f'total_acc_z_{split}.txt'))
            gx = np.loadtxt(os.path.join(uci_root, split, 'Inertial Signals', f'body_gyro_x_{split}.txt'))
            gy = np.loadtxt(os.path.join(uci_root, split, 'Inertial Signals', f'body_gyro_y_{split}.txt'))
            gz = np.loadtxt(os.path.join(uci_root, split, 'Inertial Signals', f'body_gyro_z_{split}.txt'))

            acc = np.dstack((ax, ay, az)) * 9.80665
            gyro = np.dstack((gx, gy, gz))
            data = np.concatenate([acc, gyro], axis=2)

            mask = np.isin(y_raw, list(target_map.keys()))
            data = data[mask]
            y = y_raw[mask]
            y = [target_map[k] for k in y]

            X_flat = data.reshape(-1, 6)[::2]  # Downsample
            y_flat = np.repeat(y, 128)[::2]

            if split == 'train':
                X_train, y_train = X_flat, y_flat
            else:
                X_test, y_test = X_flat, y_flat
        except:
            pass
    if 'X_train' in locals():
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        np.save(os.path.join(SAVE_DIR, 'source_features_uci_raw.npy'), {'X': X, 'y': y})
        print(f" UCI Loaded. Shape: {X.shape}")


def load_usc():

    print("\n[4/4] USC (Hip) -> Loading...")
    usc_root = None
    files = glob.glob(os.path.join(DATA_ROOT, '**', 'Subject1', '*.mat'), recursive=True)
    if files:
        usc_root = os.path.dirname(os.path.dirname(files[0]))
    elif os.path.exists(os.path.join(DATA_ROOT, 'USC-HAD')):
        usc_root = os.path.join(DATA_ROOT, 'USC-HAD')
    if not usc_root: print("USC Not Found"); return

    id_map = {1: 0, 4: 1, 5: 2, 10: 3}
    all_data, all_labels = [], []
    for root, dirs, files in os.walk(usc_root):
        for f in files:
            if f.endswith('.mat') and f.startswith('a') and 't' in f:
                try:
                    aid = int(f.split('t')[0][1:])
                    if aid in id_map:
                        full_path = os.path.join(root, f)
                        mat = sio.loadmat(full_path)
                        if 'sensor_readings' in mat:
                            raw = mat['sensor_readings']
                            acc = raw[:, 0:3] * 9.80665
                            gyro = raw[:, 3:6]
                            d = np.hstack([acc, gyro])[::4]
                            all_data.append(d)
                            all_labels.append(np.full(len(d), id_map[aid]))
                except:
                    pass
    if all_data:
        X, y = np.vstack(all_data), np.concatenate(all_labels)
        np.save(os.path.join(SAVE_DIR, 'source_features_usc_raw.npy'), {'X': X, 'y': y})
        print(f"USC Loaded. Shape: {X.shape}")


if __name__ == '__main__':
    load_dsads()
    load_pamap2()
    load_uci()
    load_usc()