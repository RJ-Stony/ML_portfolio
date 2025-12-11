import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import event2idx, result2idx, simplify_event, simplify_result

class EpisodeHybridDataset(Dataset):
    """
    [Hybrid Dataset 클래스]
    - 딥러닝 모델에 주입할 데이터를 정의하고 가공하는 역할
    - 시계열 데이터(LSTM용)와 요약 정보(MLP용)를 함께 제공함
    
    1. 시퀀스 데이터: 선수들의 움직임, 패스 등 시간 순서가 있는 12가지 정보
    2. 에피소드 피처: 이 공격 시도의 전체적인 특징 (패스 비율, 총 이동거리 등)
    """
    def __init__(self, df_events, df_episode, max_len=270):
        self.max_len = max_len
        
        # 경기 에피소드(공격 시도)별로 데이터 그룹화
        # -> 하나의 공격 작업을 하나의 샘플로 만들기 위함
        self.groups = list(df_events.groupby("game_episode"))
        
        # 에피소드 ID로 요약 피처를 빠르게 찾기 위해 인덱스 설정
        self.epi_feat_map = df_episode.set_index("game_episode")
        
        # feature_engineering.py에서 만든 7가지 요약 통계량 컬럼명
        self.epi_feat_cols = [
            "epi_len", "ratio_pass", "ratio_carry", 
            "cum_dx", "cum_dy", "movement_norm", "angle_std"
        ]

    def __len__(self):
        # 전체 데이터셋의 샘플 개수 반환
        return len(self.groups)

    def __getitem__(self, idx):
        # idx번째 에피소드 데이터를 가져와서 텐서(Tensor)로 변환
        game_episode, g = self.groups[idx]
        
        # 1. 시퀀스 데이터 전처리 (시간 흐름에 따른 데이터)
        g = g.sort_values(["time_seconds", "action_id"])
        
        # (1) 기본 좌표 및 시간 정보 추출
        sx = g["start_x"].values
        sy = g["start_y"].values
        t = g["time_seconds"].values
        
        # (2) 변화량 계산 (속도 및 방향 정보)
        # 이전 위치와의 차이를 구해 선수가 얼마나 빠르게, 어느 방향으로 움직였는지 파악
        dx = g["start_x"].diff().fillna(0).values
        dy = g["start_y"].diff().fillna(0).values
        dist = np.sqrt(dx**2 + dy**2) # 이동 거리
        angle = np.arctan2(dy, dx)    # 이동 방향 (라디안 각도)
        
        # (3) 시간 차이 (이벤트 간 간격)
        dt = g["time_seconds"].diff().fillna(0).values
        
        # (4) 누적 이동량 (공격 시작부터 현재까지 총 이동)
        # 공격이 얼마나 진행되었는지 문맥 파악 용도
        cum_dx = np.cumsum(dx)
        cum_dy = np.cumsum(dy)
        move_norm = np.sqrt(cum_dx**2 + cum_dy**2)
        
        # (5) 상대적 위치 및 시간 정보 (0~1 사이 값)
        # 공격의 초반부인지 후반부인지 모델에게 힌트 제공
        T = len(g)
        step_idx_norm = np.arange(T) / (T - 1) if T > 1 else np.zeros(T)
        
        t_min, t_max = t.min(), t.max()
        time_rel = (t - t_min) / (t_max - t_min) if t_max > t_min else np.zeros(T)

        # (6) 데이터 정규화 (Normalization)
        # 경기장 크기(105x68) 등 큰 숫자를 0~1 근처의 작은 숫자로 변환
        # -> 딥러닝 모델 학습 속도 향상 및 안정적 수렴 유도
        sx_n = sx / 105.0
        sy_n = sy / 68.0
        dx_n = dx / 40.0        # 한 번에 40m 이상 이동하는 경우는 드물어서 40으로 나눔
        dy_n = dy / 40.0
        dist_n = dist / 40.0
        angle_n = angle / np.pi # 각도는 -pi ~ pi 범위이므로 pi로 나눔
        dt_n = np.clip(dt / 3.0, 0, 1) # 이벤트 간격 3초 넘으면 그냥 1로 통일 (Clip)
        cum_dx_n = cum_dx / 60.0
        cum_dy_n = cum_dy / 60.0
        move_n = move_norm / 60.0
        
        # (7) 12가지 특징을 하나로 합침 (Feature Stacking) -> [시간길이, 12]
        seq = np.stack([
            sx_n, sy_n, dx_n, dy_n, dist_n, angle_n, dt_n,
            cum_dx_n, cum_dy_n, move_n, step_idx_norm, time_rel
        ], axis=1).astype("float32")
        
        # (8) 범주형 데이터(이벤트 종류, 결과) 처리
        # 문자열을 미리 약속된 정수 인덱스로 변환 (Pass -> 8, Success -> 2 등)
        if "event_s" in g.columns:
            ev = g["event_s"].map(event2idx).fillna(event2idx["Misc"]).values.astype("int64")
        else:
            ev = g["type_name"].apply(simplify_event).map(event2idx).fillna(event2idx["Misc"]).values.astype("int64")
            
        if "result_s" in g.columns:
            rs = g["result_s"].map(result2idx).fillna(result2idx["None"]).values.astype("int64")
        else:
            rs = g["result_name"].apply(simplify_result).map(result2idx).fillna(result2idx["None"]).values.astype("int64")

        # 2. 에피소드 요약 정보 가져오기 (전역 문맥)
        # Feature Engineering 단계에서 미리 구해둔 통계값들을 조회
        if game_episode in self.epi_feat_map.index:
            row = self.epi_feat_map.loc[game_episode]
            cluster_id = int(row["cluster_id"]) # 공격 스타일 그룹 ID
            epi_feats = row[self.epi_feat_cols].values.astype("float32")
        else:
            # 정보가 없는 경우 0으로 채움 (예외 처리)
            cluster_id = 0
            epi_feats = np.zeros(len(self.epi_feat_cols), dtype="float32")

        # 3. 정답 데이터 (Target)
        # 모델이 맞춰야 할 최종 패스의 도착 좌표 (0~1 정규화)
        target_x = g["end_x"].values[-1] / 105.0
        target_y = g["end_y"].values[-1] / 68.0
        target = np.array([target_x, target_y], dtype="float32")

        return (torch.tensor(seq), 
                torch.tensor(ev), 
                torch.tensor(rs), 
                torch.tensor(cluster_id), 
                torch.tensor(epi_feats), 
                torch.tensor(target))

def collate_hybrid(batch):
    """
    [배치 데이터 처리 함수]
    - 여러 개의 샘플을 묶어서 하나의 배치(Batch)로 만듦
    - 샘플마다 시퀀스 길이(이벤트 개수)가 다르므로, 길이를 맞추는 작업(Padding) 수행
    """
    seqs, evs, rss, clusters, epi_feats, targets = zip(*batch)
    
    # 각 샘플의 원래 길이 기록 (나중에 LSTM이 패딩 무시할 때 사용)
    lengths = torch.tensor([len(s) for s in seqs])
    
    # 패딩(Padding): 가장 긴 샘플 길이에 맞춰 짧은 샘플들의 뒷부분을 0으로 채움
    # batch_first=True: (배치크기, 시간, 특징) 순서로 정렬
    seqs_pad = pad_sequence(seqs, batch_first=True)
    evs_pad = pad_sequence(evs, batch_first=True, padding_value=0)
    rss_pad = pad_sequence(rss, batch_first=True, padding_value=0)
    
    # 나머지 데이터들은 길이가 고정되어 있으므로 단순히 쌓음(Stack)
    clusters = torch.stack(clusters)
    epi_feats = torch.stack(epi_feats)
    targets = torch.stack(targets)
    
    return seqs_pad, evs_pad, rss_pad, lengths, clusters, epi_feats, targets