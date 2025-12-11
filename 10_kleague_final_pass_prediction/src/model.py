import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import event2idx, result2idx

class FinalPassLSTMWithLastK(nn.Module):
    def __init__(self, num_feats=12, event_emb_dim=6, result_emb_dim=3, hidden_dim=96, k_last=3, num_layers=1):
        super().__init__()
        self.event_emb = nn.Embedding(len(event2idx), event_emb_dim)
        self.result_emb = nn.Embedding(len(result2idx), result_emb_dim)
        input_dim = num_feats + event_emb_dim + result_emb_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.k_last = k_last
        self.lastk_mlp = nn.Sequential(
            nn.Linear(hidden_dim * k_last, hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, seq, ev, rs, lengths):
        ev_e = self.event_emb(ev)
        rs_e = self.result_emb(rs)
        x = torch.cat([seq, ev_e, rs_e], dim=-1)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)

        h_context = h_n[-1]
        B, T, H = out_padded.size()
        lastk_list = []

        for i in range(B):
            L = lengths[i].item()
            end = L
            start = max(0, end - self.k_last)
            lastk = out_padded[i, start:end]

            if lastk.size(0) < self.k_last:
                pad = torch.zeros(self.k_last - lastk.size(0), H, device=seq.device)
                lastk = torch.cat([pad, lastk], dim=0)

            lastk_list.append(lastk.reshape(-1))

        lastk_tensor = torch.stack(lastk_list)
        h_lastk = self.lastk_mlp(lastk_tensor)
        h = torch.cat([h_context, h_lastk], dim=1)
        out = self.fc(h)
        return out