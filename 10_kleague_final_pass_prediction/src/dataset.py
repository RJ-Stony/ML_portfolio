import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import simplify_event, simplify_result, event2idx, result2idx

"""
- 수집된 축구 경기 데이터(DataFrame)를 모델이 학습할 수 있는 형태(Tensor)로 변환
1. build_episode_sequence: 하나의 득점 시퀀스(에피소드)를 받아 Feature들 추출
2. EpisodeDataset: PyTorch DataLoader에서 사용할 수 있는 Dataset 클래스
3. collate_fn: 길이가 다른 시퀀스들을 배치(Batch)로 묶을 때 패딩(Padding) 처리
"""

def build_episode_sequence(g: pd.DataFrame, for_train: bool = True):
    """
    하나의 게임 에피소드(g)를 입력받아 모델 입력용 Feature들 생성
    
    Args:
        g (pd.DataFrame): 한 에피소드(골이 나오기 전까지의 일련의 이벤트들)의 데이터프레임
        for_train (bool): 학습용이면 타겟값(Goal 좌표)을 반환하고, 아니면 None 반환
        
    Returns:
        feats (np.array): (L, 12) 형태의 수치형 특징 벡터들
        event_idx (np.array): (L,) 형태의 이벤트 타입 인덱스
        result_idx (np.array): (L,) 형태의 결과 타입 인덱스
        target (np.array): (2,) 형태의 정답 좌표 (x, y) - 학습용일 때만 존재
    """
    # 인덱스 초기화 및 복사 (원본 데이터 보존)
    g = g.reset_index(drop=True).copy()
    
    # 데이터가 너무 짧으면(2개 미만) 시퀀스로서 의미가 없으므로 무시
    if len(g) < 2:
        return None, None, None, None

    # 1. 위치 및 시간 정보 추출
    sx = g["start_x"].values  # 시작 x 좌표
    sy = g["start_y"].values  # 시작 y 좌표
    t  = g["time_seconds"].values  # 경기 시간(초)

    # 2. 변화량 계산 - 속도나 방향성을 알기 위함
    # dx, dy: 이전 위치와의 차이 (이동 거리)
    dx = np.diff(sx, prepend=sx[0])
    dy = np.diff(sy, prepend=sy[0])
    dist = np.sqrt(dx**2 + dy**2)  # 이동 거리(유클리드 거리)
    angle = np.arctan2(dy, dx)     # 이동 각도 (라디안)

    # dt: 이전 이벤트와의 시간 차이
    dt = np.diff(t, prepend=t[0])
    dt[dt < 0] = 0  # 시간 차이는 음수가 될 수 없으므로 0으로 클리핑

    # 3. 누적 이동량 계산 - 에피소드 시작부터 현재까지 얼마나 이동했는지
    cum_dx = np.cumsum(dx)
    cum_dy = np.cumsum(dy)
    move_norm = np.sqrt(cum_dx**2 + cum_dy**2) # 시작점으로부터의 직선 거리

    # 4. 시퀀스 내의 상대적 위치 계산
    T = len(g)
    step_idx = np.arange(T)
    # 0~1 사이로 정규화 (초반인지 후반인지 정보를 주기 위함)
    step_idx_norm = step_idx / (T - 1) if T > 1 else np.zeros(T)

    # 5. 시간 정규화 (에피소드 내에서 흐른 시간 비율)
    t_min, t_max = t.min(), t.max()
    time_rel = (t - t_min) / (t_max - t_min) if t_max > t_min else np.zeros(T)

    # 6. 정규화(Norm) 수행
    # 좌우 105m, 상하 68m 경기장 크기를 기준으로 0~1 근처 값으로 스케일링
    sx_norm = sx / 105.0
    sy_norm = sy / 68.0
    dx_norm = dx / 40.0       # EDA 결과상, 한 번 이동 시 40m 이상 이동하는 경우는 드물기 때문에 40m로 클리핑
    dy_norm = dy / 40.0
    dist_norm = dist / 40.0
    angle_norm = angle / np.pi          # 각도는 -pi ~ pi 사이이므로 pi로 나눔
    dt_norm = np.clip(dt / 3.0, 0, 1)   # EDA 결과상, 이벤트 간 간격은 보통 3초로, 해당 기준으로 클리핑
    cum_dx_norm = cum_dx / 60.0
    cum_dy_norm = cum_dy / 60.0
    move_norm_norm = move_norm / 60.0

    # 7. 수치형 피쳐 합치기 (Feature Stacking) -> (Time, 12)
    feats = np.stack([
        sx_norm, sy_norm, dx_norm, dy_norm, dist_norm,
        angle_norm, dt_norm, cum_dx_norm, cum_dy_norm,
        move_norm_norm, step_idx_norm, time_rel
    ], axis=1).astype("float32")

    # 8. 범주형 데이터(이벤트 타입) 처리
    # 이미 'event_s' 컬럼이 있으면 바로 쓰고, 없으면 simplify_event로 변환 후 인덱싱
    if "event_s" in g.columns:
        event_idx = g["event_s"].apply(lambda x: event2idx[x]).values.astype("int64")
    else:
        tmp_event = g["type_name"].astype(str).apply(simplify_event)
        event_idx = tmp_event.apply(lambda x: event2idx[x]).values.astype("int64")

    # 9. 범주형 데이터(결과 타입) 처리
    if "result_s" in g.columns:
        result_idx = g["result_s"].apply(lambda x: result2idx[x]).values.astype("int64")
    else:
        tmp_result = g["result_name"].astype(str).apply(simplify_result)
        result_idx = tmp_result.apply(lambda x: result2idx[x]).values.astype("int64")

    # 10. 정답(Target) 추출 (학습용인 경우에만)
    # 마지막 이벤트 이후 패스가 도달한 좌표(end_x, end_y)가 맞혀야 할 정답
    target = None

    if for_train:
        ex = g["end_x"].values[-1] / 105.0  # 정답도 0~1 사이로 정규화
        ey = g["end_y"].values[-1] / 68.0
        target = np.array([ex, ey], dtype="float32")

    return feats, event_idx, result_idx, target

class EpisodeDataset(Dataset):
    """
    PyTorch의 Dataset 클래스를 상속받아, 
    데이터 로더가 인덱스(idx)로 데이터에 접근할 수 있게 해줍니다.
    """
    def __init__(self, seqs, evs, rss, tgts):
        # seqs: 수치형 피쳐 시퀀스 리스트
        # evs: 이벤트 인덱스 시퀀스 리스트
        # rss: 결과 인덱스 시퀀스 리스트
        # tgts: 정답(좌표) 리스트
        self.seqs = seqs
        self.evs = evs
        self.rss = rss
        self.tgts = tgts

    def __len__(self):
        # 전체 데이터 개수(에피소드 개수) 반환
        return len(self.seqs)

    def __getitem__(self, idx):
        # idx 번째 데이터(에피소드)를 찾아 텐서(Tensor)로 변환하여 반환
        seq = torch.tensor(self.seqs[idx])
        ev  = torch.tensor(self.evs[idx])
        rs  = torch.tensor(self.rss[idx])
        tgt = torch.tensor(self.tgts[idx])
        # seq.size(0)은 시퀀스의 길이(Time step 수)
        return seq, ev, rs, seq.size(0), tgt

def collate_fn(batch):
    """
    배치(Batch) 처리를 위한 함수
    데이터마다 시퀀스 길이가 다르기 때문에, 이를 맞춰주기 위해 패딩(Padding) 사용
    """
    # batch는 (seq, ev, rs, lengths, tgt) 튜플들의 리스트
    seqs, evs, rss, lengths, tgts = zip(*batch)
    
    lengths = torch.tensor(lengths)
    tgts = torch.stack(tgts) # 정답(좌표)은 그냥 쌓으면 됨
    
    # pad_sequence: 길이가 다른 시퀀스 뒤에 0을 채워서 길이 맞추기
    # batch_first=True -> (Batch, Time, Feature) 형태로 반환
    seqs_p = pad_sequence(seqs, batch_first=True)
    evs_p  = pad_sequence(evs, batch_first=True, padding_value=0)
    rss_p  = pad_sequence(rss, batch_first=True, padding_value=0)
    
    return seqs_p, evs_p, rss_p, lengths, tgts