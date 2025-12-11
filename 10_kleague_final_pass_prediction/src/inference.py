import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
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
    # 1. Feature Engineering (테스트용 하이브리드 피처 준비)
    # ==========================================
    # LSTM 입력 외에 추가로 사용할 '에피소드 요약 정보'가 있는지 확인
    test_epi_path = "../Data/test_episode_features.csv"
    
    # 만약 피처 파일이 없으면 생성 (시간이 좀 걸릴 수 있음)
    if not os.path.exists(test_epi_path):
        print("Test Hybrid Features not found. Generating now...")
        feature_engineering.generate_hybrid_features(
            train_path=config.TRAIN_PATH,
            test_path=config.TEST_PATH, 
            save_path="../Data/"
        )
    else:
        print("Test Hybrid Features found. Loading...")

    # ==========================================
    # 2. Data Load (데이터 불러오기)
    # ==========================================
    print("Loading Test Event Data...")
    
    # (1) 개별 테스트 파일들을 하나로 합친 DataFrame 로드
    # feature_engineering 모듈에 있는 함수 재사용 (코드 중복 방지)
    df_test_events = feature_engineering.load_test_data(config.TEST_PATH)
    
    # 전처리 (Train과 동일하게 적용해야 모델이 헷갈리지 않음)
    df_test_events["event_s"] = df_test_events["type_name"].astype(str).apply(simplify_event)
    df_test_events["result_s"] = df_test_events["result_name"].astype(str).apply(simplify_result)
    
    # (2) 에피소드 Hybrid 피처 로드 (요약 통계 정보)
    df_test_epi_features = pd.read_csv(test_epi_path)
    
    # ==========================================
    # 3. Dataset & DataLoader (데이터 주입기 생성)
    # ==========================================
    # Train과 동일한 Dataset 클래스 사용 -> 전처리 방식 통일
    test_dataset = EpisodeHybridDataset(df_test_events, df_test_epi_features)
    
    # [중요] Shuffle=False: 테스트는 정답 제출 순서가 중요하므로 절대 섞으면 안 됨
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_hybrid)
    
    # ==========================================
    # 4. Model Load (모델 불러오기)
    # ==========================================
    # 모델 구조를 코드로 먼저 생성 (껍데기 만들기)
    model = HybridFinalPassLSTM(
        num_feats=12,        # Dataset 출력과 일치
        event_emb_dim=6,
        result_emb_dim=3,
        cluster_emb_dim=4,
        epi_feat_dim=7,      
        hidden_dim=128,      
        num_layers=2         
    ).to(config.DEVICE)
    
    # 학습된 가중치(파라미터) 파일이 있는지 확인
    if not os.path.exists(config.SAVE_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {config.SAVE_MODEL_PATH}. Train first!")
        
    # 가중치를 모델에 덮어씌움
    model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=config.DEVICE))
    
    # 평가 모드 전환 (Dropout 등 학습 전용 기능 끄기)
    model.eval() 
    
    print("Model loaded successfully.")

    # ==========================================
    # 5. Inference Loop (실제 추론 수행)
    # ==========================================
    results = [] # 예측 결과를 담을 리스트
    
    print("Starting Inference...")
    
    with torch.no_grad(): # 기록 안 함 (메모리 절약)
        for batch in tqdm(test_loader, desc="Predicting"):
            # collate_hybrid가 반환하는 순서대로 데이터 받기
            seq, ev, rs, lengths, cluster, epi_feat, _ = batch
            
            # GPU로 이동
            seq = seq.to(config.DEVICE)
            ev = ev.to(config.DEVICE)
            rs = rs.to(config.DEVICE)
            lengths = lengths.to(config.DEVICE)
            cluster = cluster.to(config.DEVICE)
            epi_feat = epi_feat.to(config.DEVICE)
            
            # 모델 예측 (Forward)
            pred = model(seq, ev, rs, lengths, cluster, epi_feat)
            
            # 결과 복원 (0~1로 압축된 값을 실제 경기장 좌표로)
            pred_np = pred.cpu().numpy()
            pred_x = pred_np[:, 0] * 105
            pred_y = pred_np[:, 1] * 68
            
            # 결과 리스트에 추가
            for x, y in zip(pred_x, pred_y):
                results.append((x, y))

    # ==========================================
    # 6. Submission File Generation (제출 파일 생성)
    # ==========================================
    # Dataset에 저장된 실제 game_episode ID 순서 가져오기
    # (DataLoader에서 shuffle=False 했으므로 순서 일치함)
    episode_ids = [g[0] for g in test_dataset.groups]
    
    # 예측 결과 DataFrame 만들기
    df_preds = pd.DataFrame({
        "game_episode": episode_ids,
        "end_x": [r[0] for r in results],
        "end_y": [r[1] for r in results]
    })
    
    # 제출 양식 파일 로드
    submission = pd.read_csv(config.SUBMISSION_PATH)
    
    # 기존 예시 값 제거하고 우리가 예측한 값으로 교체 (Merge 사용)
    if "end_x" in submission.columns:
        submission = submission.drop(columns=["end_x", "end_y"])
        
    final_submission = submission.merge(df_preds, on="game_episode", how="left")
    
    # 혹시 매칭 안 된 에피소드가 있으면 0으로 채움 (안전장치)
    if final_submission.isnull().sum().sum() > 0:
        print("WARNING: Some episodes are missing predictions. Filling with 0.")
        final_submission = final_submission.fillna(0)
    
    # 최종 파일 저장
    save_path = "../Data/hybrid_bilstm_submission.csv"
    final_submission.to_csv(save_path, index=False)
    
    print(f"Inference Done! Submission saved to: {save_path}")

if __name__ == "__main__":
    main()