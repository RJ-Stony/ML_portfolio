import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os

import config
from utils import simplify_event, simplify_result
from dataset import build_episode_sequence
from model import FinalPassLSTMWithLastK

def load_all_test_files(test_meta, base_dir="../Data"):
    cache = {}
    paths = test_meta["path"].unique()
    for path in tqdm(paths, desc="Loading test files"):
        clean = path[1:]
        full_path = os.path.join(base_dir, clean)
        try:
            df = pd.read_csv(full_path)
            cache[path] = df
        except FileNotFoundError:
            pass
    return cache

def main():
    test_meta = pd.read_csv(config.TEST_PATH)
    submission = pd.read_csv(config.SUBMISSION_PATH)
    submission = submission.merge(test_meta, on="game_episode", how="left")

    model = FinalPassLSTMWithLastK(hidden_dim=config.HIDDEN_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    file_cache = load_all_test_files(test_meta, base_dir="../Data")

    preds_x, preds_y = [], []

    with torch.no_grad():
        for _, row in tqdm(submission.iterrows(), total=len(submission), desc="Inference"):
            path = row["path"]
            if path not in file_cache:
                preds_x.append(0)
                preds_y.append(0)
                continue

            g = file_cache[path].copy()

            g["event_s"] = g["type_name"].astype(str).apply(simplify_event)
            g["result_s"] = g["result_name"].astype(str).apply(simplify_result)

            seq, ev, rs, _ = build_episode_sequence(g, for_train=False)
            
            if seq is None:
                preds_x.append(0)
                preds_y.append(0)
                continue

            seq = torch.tensor(seq).unsqueeze(0).to(config.DEVICE)
            ev  = torch.tensor(ev).unsqueeze(0).to(config.DEVICE)
            rs  = torch.tensor(rs).unsqueeze(0).to(config.DEVICE)
            L   = torch.tensor([seq.shape[1]]).to(config.DEVICE)

            pred = model(seq, ev, rs, L)[0].cpu().numpy()

            preds_x.append(pred[0] * 105)
            preds_y.append(pred[1] * 68)

    submission["end_x"] = preds_x
    submission["end_y"] = preds_y
    
    save_path = "../Data/step4_submit.csv"
    submission[["game_episode", "end_x", "end_y"]].to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()