import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from utils import simplify_event, simplify_result
from dataset import build_episode_sequence, EpisodeDataset, collate_fn
from model import FinalPassLSTMWithLastK

def main():
    df = pd.read_csv(config.TRAIN_PATH)
    df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)
    
    df["event_s"] = df["type_name"].astype(str).apply(simplify_event)
    df["result_s"] = df["result_name"].astype(str).apply(simplify_result)

    episodes, events, results, targets = [], [], [], []
    for _, g in tqdm(df.groupby("game_episode"), desc="Processing Episodes"):
        seq, ev, rs, tgt = build_episode_sequence(g)
        if seq is None: continue
        episodes.append(seq)
        events.append(ev)
        results.append(rs)
        targets.append(tgt)

    idx_train, idx_valid = train_test_split(np.arange(len(episodes)), test_size=0.2, random_state=42)

    train_loader = DataLoader(
        EpisodeDataset([episodes[i] for i in idx_train], [events[i] for i in idx_train],
                       [results[i] for i in idx_train], [targets[i] for i in idx_train]),
        batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        EpisodeDataset([episodes[i] for i in idx_valid], [events[i] for i in idx_valid],
                       [results[i] for i in idx_valid], [targets[i] for i in idx_valid]),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = FinalPassLSTMWithLastK(hidden_dim=config.HIDDEN_DIM).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.MSELoss()

    best_dist = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0
        for seq, ev, rs, lengths, tgt in train_loader:
            seq, ev, rs, lengths, tgt = seq.to(config.DEVICE), ev.to(config.DEVICE), rs.to(config.DEVICE), lengths.to(config.DEVICE), tgt.to(config.DEVICE)
            
            optimizer.zero_grad()
            pred = model(seq, ev, rs, lengths)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)
        
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        dists = []
        with torch.no_grad():
            for seq, ev, rs, lengths, tgt in valid_loader:
                seq, ev, rs, lengths, tgt = seq.to(config.DEVICE), ev.to(config.DEVICE), rs.to(config.DEVICE), lengths.to(config.DEVICE), tgt.to(config.DEVICE)
                pred = model(seq, ev, rs, lengths)
                
                pred_np = pred.cpu().numpy()
                tgt_np = tgt.cpu().numpy()
                
                px, py = pred_np[:, 0] * 105, pred_np[:, 1] * 68
                tx, ty = tgt_np[:, 0] * 105, tgt_np[:, 1] * 68
                
                dists.append(np.sqrt((px - tx)**2 + (py - ty)**2))
        
        mean_dist = np.concatenate(dists).mean()
        print(f"[Epoch {epoch}] Loss: {train_loss:.4f} | Mean Dist: {mean_dist:.4f}")

        if mean_dist < best_dist:
            best_dist = mean_dist
            torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
            print(f"--> Best model saved (Dist: {best_dist:.4f})")

if __name__ == "__main__":
    main()