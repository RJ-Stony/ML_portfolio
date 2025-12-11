import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 사용자 정의 모듈 로드
import config
from model import HybridFinalPassLSTM
from dataset import EpisodeHybridDataset, collate_hybrid
from utils import simplify_event, simplify_result
import feature_engineering

def main():
    print(f"Using Device: {config.DEVICE}")

    # ==========================================
    # 1. Feature Engineering (하이브리드 피처 준비)
    # ==========================================
    # LSTM 입력 외에 추가로 사용할 '에피소드 요약 정보'가 있는지 확인
    # 없으면 feature_engineering.py를 실행해서 파일 생성
    real_epi_path = "../Data/train_episode_features.csv"
    
    if not os.path.exists(real_epi_path):
        print("Hybrid Features not found. Generating now...")
        feature_engineering.generate_hybrid_features(
            train_path=config.TRAIN_PATH,
            test_path=config.TEST_PATH, 
            save_path="../Data/"
        )
    else:
        print("Hybrid Features found. Loading...")

    # ==========================================
    # 2. Data Load & Merge (데이터 불러오기)
    # ==========================================
    # (1) 이벤트 데이터 (시퀀스용)
    # 선수들의 세세한 움직임 로그가 담긴 파일 로드
    df_train = pd.read_csv(config.TRAIN_PATH)
    
    # 시간 순서가 섞여있을 수 있으므로 정렬 (아주 중요)
    df_train = df_train.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)
    
    # 복잡한 이벤트 명칭을 단순화 (예: Pass_Corner -> Pass)
    df_train["event_s"] = df_train["type_name"].astype(str).apply(simplify_event)
    df_train["result_s"] = df_train["result_name"].astype(str).apply(simplify_result)
    
    # (2) 에피소드 데이터 (요약 정보용)
    # 위에서 만든 통계/클러스터링 피처 로드
    df_epi_features = pd.read_csv(real_epi_path)
    
    # ==========================================
    # 3. Train / Validation Split (데이터 나누기)
    # ==========================================
    # [Data Leakage 방지]
    # 단순히 랜덤으로 섞으면 같은 경기의 앞부분은 학습하고 뒷부분은 검증하게 되어,
    # 모델이 정답을 미리 외워버리는 문제가 생김.
    # 이를 막기 위해 '에피소드 ID'를 기준으로 통째로 나눔.
    all_episodes = df_train["game_episode"].unique()
    train_epis, valid_epis = train_test_split(all_episodes, test_size=0.2, random_state=42)
    
    print(f"Train Episodes: {len(train_epis)}")
    print(f"Valid Episodes: {len(valid_epis)}")
    
    # 나뉜 ID에 해당하는 이벤트 데이터만 추출
    train_events = df_train[df_train["game_episode"].isin(train_epis)]
    valid_events = df_train[df_train["game_episode"].isin(valid_epis)]
    
    # ==========================================
    # 4. Dataset & DataLoader (데이터 주입기 생성)
    # ==========================================
    # Dataset: 데이터를 하나씩 텐서로 변환해주는 역할
    train_dataset = EpisodeHybridDataset(train_events, df_epi_features)
    valid_dataset = EpisodeHybridDataset(valid_events, df_epi_features)
    
    # DataLoader: 데이터를 배치 단위로 묶어서 모델에 공급
    # collate_hybrid: 길이가 다른 시퀀스를 패딩으로 맞춰주는 함수
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_hybrid=collate_hybrid)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_hybrid=collate_hybrid)
    
    # ==========================================
    # 5. Model Initialization (모델 생성)
    # ==========================================
    model = HybridFinalPassLSTM(
        num_feats=12,        # 시퀀스 데이터 특징 개수 (좌표, 속도 등)
        event_emb_dim=6,
        result_emb_dim=3,
        cluster_emb_dim=4,
        epi_feat_dim=7,      # 통계 요약 피처 개수
        hidden_dim=128,      # LSTM 내부 뉴런 개수
        num_layers=2         
    ).to(config.DEVICE) # GPU로 이동
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR) # 최적화 알고리즘 (Adam)
    criterion = nn.MSELoss() # 손실 함수 (거리 오차 최소화 목적)
    
    # ==========================================
    # 6. Training Loop (학습 반복)
    # ==========================================
    best_dist = float("inf") # 최고 기록 저장용 (낮을수록 좋음)
    patience_limit = 5       # 조기 종료(Early Stopping) 카운트
    patience_counter = 0
    
    print("\n🚀 Start Training Hybrid LSTM...")
    
    for epoch in range(1, config.EPOCHS + 1):
        # [학습 모드]
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        
        # 배치 데이터 반복 (collate_hybrid 함수가 반환하는 순서대로 받음)
        for seq, ev, rs, lengths, cluster, epi_feat, target in progress_bar:
            # 모든 데이터를 GPU로 이동
            seq = seq.to(config.DEVICE)
            ev = ev.to(config.DEVICE)
            rs = rs.to(config.DEVICE)
            lengths = lengths.to(config.DEVICE)
            cluster = cluster.to(config.DEVICE)
            epi_feat = epi_feat.to(config.DEVICE)
            target = target.to(config.DEVICE)
            
            # 1. 기울기 초기화
            optimizer.zero_grad()
            
            # 2. 모델 예측 (Forward)
            pred = model(seq, ev, rs, lengths, cluster, epi_feat)
            
            # 3. 오차 계산 (Loss)
            loss = criterion(pred, target)
            
            # 4. 역전파 (Backward) 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * seq.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss = total_loss / len(train_loader.dataset)
        
        # ==========================================
        # 7. Validation (검증 모드)
        # ==========================================
        model.eval()
        dists = []
        
        with torch.no_grad(): # 검증 때는 학습 안 함 (메모리 절약)
            for seq, ev, rs, lengths, cluster, epi_feat, target in valid_loader:
                seq = seq.to(config.DEVICE)
                ev = ev.to(config.DEVICE)
                rs = rs.to(config.DEVICE)
                lengths = lengths.to(config.DEVICE)
                cluster = cluster.to(config.DEVICE)
                epi_feat = epi_feat.to(config.DEVICE)
                target = target.to(config.DEVICE)
                
                pred = model(seq, ev, rs, lengths, cluster, epi_feat)
                
                # 정규화된 좌표(0~1)를 실제 경기장 좌표(105x68)로 변환
                pred_np = pred.cpu().numpy()
                tgt_np = target.cpu().numpy()
                
                px, py = pred_np[:, 0] * 105, pred_np[:, 1] * 68
                tx, ty = tgt_np[:, 0] * 105, tgt_np[:, 1] * 68
                
                # 유클리드 거리 오차 계산
                batch_dists = np.sqrt((px - tx)**2 + (py - ty)**2)
                dists.extend(batch_dists)
        
        mean_dist = np.mean(dists)
        print(f"\t[Result] Train Loss: {train_loss:.4f} | Valid Mean Dist: {mean_dist:.4f}m")
        
        # ==========================================
        # 8. Checkpoint & Early Stopping (저장 및 종료)
        # ==========================================
        # 지금까지 본 것 중 가장 성능이 좋으면 저장
        if mean_dist < best_dist:
            best_dist = mean_dist
            patience_counter = 0
            torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
            print(f"\tBest model saved! (Dist: {best_dist:.4f})")
        else:
            # 성능이 안 좋아지면 카운트 증가
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch}")
                break

if __name__ == "__main__":
    main()