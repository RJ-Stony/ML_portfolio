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

"""
- 모델 학습시키는 메인 스크립트
1. 데이터 로드하고 전처리하여 EpisodeDataset 만듦
2. 모델 초기화하고, 손실 함수와 최적화 함수 설정
3. Epoch 반복하며 모델 학습하고, 매 Epoch마다 검증(Validation) 수행
4. 검증 성능이 가장 좋은 모델 저장
"""

def main():
    # 1. 학습 데이터 로드
    df = pd.read_csv(config.TRAIN_PATH)
    # 에피소드(경기) 별, 그리고 시간 순서대로 정렬
    df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)
    
    # 2. 전처리: 이벤트 이름과 결과 단순화
    df["event_s"] = df["type_name"].astype(str).apply(simplify_event)
    df["result_s"] = df["result_name"].astype(str).apply(simplify_result)

    # 3. 에피소드 데이터 구축
    # DataFrame 순회하며 모델에 들어갈 시퀀스 데이터로 변환
    episodes, events, results, targets = [], [], [], []
    for _, g in tqdm(df.groupby("game_episode"), desc="Processing Episodes"):
        seq, ev, rs, tgt = build_episode_sequence(g)
        if seq is None: continue # 데이터 유효하지 않으면 건너뜀
        episodes.append(seq)
        events.append(ev)
        results.append(rs)
        targets.append(tgt)

    # 4. 학습/검증 데이터 분리 (8:2 비율)
    idx_train, idx_valid = train_test_split(np.arange(len(episodes)), test_size=0.2, random_state=42)

    # 5. 데이터 로더 생성
    # 학습 데이터 로더 (Shuffle=True: 데이터 섞어서 학습)
    train_loader = DataLoader(
        EpisodeDataset([episodes[i] for i in idx_train], [events[i] for i in idx_train],
                       [results[i] for i in idx_train], [targets[i] for i in idx_train]),
        batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # 검증 데이터 로더 (Shuffle=False: 고정된 순서로 검증)
    valid_loader = DataLoader(
        EpisodeDataset([episodes[i] for i in idx_valid], [events[i] for i in idx_valid],
                       [results[i] for i in idx_valid], [targets[i] for i in idx_valid]),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 6. 모델, 최적화기(Optimizer), 손실 함수(Loss) 설정
    model = FinalPassLSTMWithLastK(hidden_dim=config.HIDDEN_DIM).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.MSELoss() # 회귀 문제(좌표 예측)이므로 MSE Loss 사용

    best_dist = float("inf") # 가장 좋은 거리 오차 기록용 (초기값 무한대)

    # 7. 학습 루프 (Training Loop)
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0
    
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        
        for seq, ev, rs, lengths, tgt in progress_bar:
            # 데이터를 GPU(또는 CPU)로 이동
            seq, ev, rs, lengths, tgt = seq.to(config.DEVICE), ev.to(config.DEVICE), rs.to(config.DEVICE), lengths.to(config.DEVICE), tgt.to(config.DEVICE)
            
            optimizer.zero_grad() # 이전 기울기 초기화
            pred = model(seq, ev, rs, lengths) # 예측
            loss = criterion(pred, tgt) # 손실 계산
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트
            
            total_loss += loss.item() * seq.size(0)
            
            # 진행 상황 표시에 현재 배치의 손실값 업데이트
            progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        dists = []

        with torch.no_grad(): # 검증 때는 기울기 계산 하지 않음 (메모리 절약)
            for seq, ev, rs, lengths, tgt in valid_loader:
                seq, ev, rs, lengths, tgt = seq.to(config.DEVICE), ev.to(config.DEVICE), rs.to(config.DEVICE), lengths.to(config.DEVICE), tgt.to(config.DEVICE)
                pred = model(seq, ev, rs, lengths)
                
                # 정규화된 좌표(0~1)를 실제 경기장 좌표로 복원해 거리 오차 계산
                # numpy() : torch tensor를 numpy array로 변환 
                pred_np = pred.cpu().numpy()
                tgt_np = tgt.cpu().numpy()
                
                px, py = pred_np[:, 0] * 105, pred_np[:, 1] * 68
                tx, ty = tgt_np[:, 0] * 105, tgt_np[:, 1] * 68
                
                dists.append(np.sqrt((px - tx)**2 + (py - ty)**2))
        
        # numpy array로 변환하고 평균 거리 계산
        mean_dist = np.concatenate(dists).mean()
        
        print(f"\t[Result] Train Loss: {train_loss:.4f} | Valid Mean Dist: {mean_dist:.4f}")

        # 8. 모델 저장 (Best Model Checkpoint)
        # 이번 Epoch의 검증 오차가 역대 최고(최소)라면 저장
        if mean_dist < best_dist:
            best_dist = mean_dist
            torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
            print(f"\t--> Best model saved (Dist: {best_dist:.4f})")

if __name__ == "__main__":
    main()