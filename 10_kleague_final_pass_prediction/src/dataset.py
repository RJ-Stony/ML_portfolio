import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import simplify_event, simplify_result, event2idx, result2idx

def build_episode_sequence(g: pd.DataFrame, for_train: bool = True):
    g = g.reset_index(drop=True).copy()
    if len(g) < 2:
        return None, None, None, None

    sx = g["start_x"].values
    sy = g["start_y"].values
    t  = g["time_seconds"].values

    dx = np.diff(sx, prepend=sx[0])
    dy = np.diff(sy, prepend=sy[0])
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    dt = np.diff(t, prepend=t[0])
    dt[dt < 0] = 0

    cum_dx = np.cumsum(dx)
    cum_dy = np.cumsum(dy)
    move_norm = np.sqrt(cum_dx**2 + cum_dy**2)

    T = len(g)
    step_idx = np.arange(T)
    step_idx_norm = step_idx / (T - 1) if T > 1 else np.zeros(T)

    t_min, t_max = t.min(), t.max()
    time_rel = (t - t_min) / (t_max - t_min) if t_max > t_min else np.zeros(T)

    sx_norm = sx / 105.0
    sy_norm = sy / 68.0
    dx_norm = dx / 40.0
    dy_norm = dy / 40.0
    dist_norm = dist / 40.0
    angle_norm = angle / np.pi
    dt_norm = np.clip(dt / 3.0, 0, 1)
    cum_dx_norm = cum_dx / 60.0
    cum_dy_norm = cum_dy / 60.0
    move_norm_norm = move_norm / 60.0

    feats = np.stack([
        sx_norm, sy_norm, dx_norm, dy_norm, dist_norm,
        angle_norm, dt_norm, cum_dx_norm, cum_dy_norm,
        move_norm_norm, step_idx_norm, time_rel
    ], axis=1).astype("float32")

    if "event_s" in g.columns:
        event_idx = g["event_s"].apply(lambda x: event2idx[x]).values.astype("int64")
    else:
        tmp_event = g["type_name"].astype(str).apply(simplify_event)
        event_idx = tmp_event.apply(lambda x: event2idx[x]).values.astype("int64")

    if "result_s" in g.columns:
        result_idx = g["result_s"].apply(lambda x: result2idx[x]).values.astype("int64")
    else:
        tmp_result = g["result_name"].astype(str).apply(simplify_result)
        result_idx = tmp_result.apply(lambda x: result2idx[x]).values.astype("int64")

    target = None
    if for_train:
        ex = g["end_x"].values[-1] / 105.0
        ey = g["end_y"].values[-1] / 68.0
        target = np.array([ex, ey], dtype="float32")

    return feats, event_idx, result_idx, target

class EpisodeDataset(Dataset):
    def __init__(self, seqs, evs, rss, tgts):
        self.seqs = seqs
        self.evs = evs
        self.rss = rss
        self.tgts = tgts

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx])
        ev  = torch.tensor(self.evs[idx])
        rs  = torch.tensor(self.rss[idx])
        tgt = torch.tensor(self.tgts[idx])
        return seq, ev, rs, seq.size(0), tgt

def collate_fn(batch):
    seqs, evs, rss, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths)
    tgts = torch.stack(tgts)
    seqs_p = pad_sequence(seqs, batch_first=True)
    evs_p  = pad_sequence(evs, batch_first=True, padding_value=0)
    rss_p  = pad_sequence(rss, batch_first=True, padding_value=0)
    return seqs_p, evs_p, rss_p, lengths, tgts