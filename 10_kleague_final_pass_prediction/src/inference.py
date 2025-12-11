import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os

import config
from utils import simplify_event, simplify_result
from dataset import build_episode_sequence
from model import FinalPassLSTMWithLastK

"""
- 학습된 모델을 불러와 실제 테스트 데이터에 대한 예측을 수행하는 코드
1. 테스트 데이터 파일들 로드
2. 저장된 모델 가중치(State Dict) 로드
3. 제출 파일(submission.csv) 각 행 순회하며 예측 수행
4. 결과 CSV 파일로 저장
"""

def load_all_test_files(test_meta, base_dir="../Data"):
    """
    테스트 메타데이터 모든 경로 파일 미리 로드하여 메모리 캐싱
    매번 파일 읽는 I/O 시간 줄일 수 있음 !!
    
    Args:
        test_meta (pd.DataFrame): 테스트 데이터 정보 메타데이터
        base_dir (str): 데이터 저장 기본 디렉토리
        
    Returns:
        dict: {파일경로: DataFrame} 형태 캐시
    """
    cache = {}
    paths = test_meta["path"].unique() # 중복되지 않는 파일 경로 리스트
    success_cnt = 0
    fail_cnt = 0
    
    print(f"DEBUG: Looking for data in {os.path.abspath(base_dir)}")
    
    for path in tqdm(paths, desc="Loading test files"):
        clean = path.replace("./", "")  # 경로 정리 (./ 제거)
        clean = os.path.normpath(clean) 
        full_path = os.path.join(base_dir, clean)
        
        try:
            df = pd.read_csv(full_path)
            cache[path] = df # 원본 path 키로 사용 (나중에 조회 편의성 위해)
            success_cnt += 1
        except FileNotFoundError:
            # 파일 못 찾았을 경우 경고 (첫 번째 케이스만 출력)
            if fail_cnt == 0:
                print(f"DEBUG(First Fail): Cannot find {os.path.abspath(full_path)}")
            fail_cnt += 1
            
    print(f"DEBUG: Successfully loaded {success_cnt} files. Failed: {fail_cnt}")
    
    if success_cnt == 0:
        print("CRITICAL WARNING: No files loaded! Please check if 'test' folder exists in '../Data'.")
        
    return cache

def main():
    # 1. 필요한 메타데이터 및 제출 양식 로드
    test_meta = pd.read_csv(config.TEST_PATH)
    submission = pd.read_csv(config.SUBMISSION_PATH)
    
    # 제출 양식에 어떤 파일의 어떤 에피소드인지 정보 합침
    submission = submission.merge(test_meta, on="game_episode", how="left")

    # 2. 모델 초기화 및 가중치 로드
    model = FinalPassLSTMWithLastK(hidden_dim=config.HIDDEN_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=config.DEVICE))
    model.eval() # 평가 모드 설정 (Dropout 등 비활성화)

    # 3. 테스트 데이터 파일 캐싱
    file_cache = load_all_test_files(test_meta, base_dir="../Data")

    preds_x, preds_y = [], []

    # 4. 추론(Inference) 수행
    with torch.no_grad():
        for _, row in tqdm(submission.iterrows(), total=len(submission), desc="Inference"):
            path = row["path"]
            # 파일이 캐시에 없으면 예측 불가 -> (0,0) 처리
            if path not in file_cache:
                preds_x.append(0)
                preds_y.append(0)
                continue

            # 해당 파일의 데이터프레임 가져오기
            g = file_cache[path].copy()

            # 전처리 (Train 때와 동일하게)
            g["event_s"] = g["type_name"].astype(str).apply(simplify_event)
            g["result_s"] = g["result_name"].astype(str).apply(simplify_result)

            # 시퀀스 데이터 생성 (for_train=False이므로 target은 None)
            seq, ev, rs, _ = build_episode_sequence(g, for_train=False)
            
            if seq is None:
                preds_x.append(0)
                preds_y.append(0)
                continue

            # 모델 입력용 텐서 변환 (Batch 차원 추가: unsqueeze)
            seq = torch.tensor(seq).unsqueeze(0).to(config.DEVICE)
            ev  = torch.tensor(ev).unsqueeze(0).to(config.DEVICE)
            rs  = torch.tensor(rs).unsqueeze(0).to(config.DEVICE)
            L   = torch.tensor([seq.shape[1]]).to(config.DEVICE) # 시퀀스 길이

            # 모델 예측
            pred = model(seq, ev, rs, L)[0].cpu().numpy()

            # 정규화된 좌표를 원래 경기장 크기로 복원
            preds_x.append(pred[0] * 105)
            preds_y.append(pred[1] * 68)

    # 5. 결과 저장
    submission["end_x"] = preds_x
    submission["end_y"] = preds_y
    
    save_path = "../Data/step4_submit.csv"
    submission[["game_episode", "end_x", "end_y"]].to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()