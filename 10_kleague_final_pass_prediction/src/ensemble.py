import pandas as pd
import numpy as np

def main():
    # 1. 방금 만든 LGBM (14.48점)
    lgbm_path = "../Data/lgbm_submission.csv"
    
    # 2. 아까 만든 Hybrid LSTM (15.7점)
    lstm_path = "../Data/hybrid_bilstm_submission.csv"
    
    # 저장할 경로
    save_path = "../Data/ensemble_submission.csv"
    
    print("Loading submissions...")
    lgbm = pd.read_csv(lgbm_path)
    lstm = pd.read_csv(lstm_path)
    
    # 에피소드 순서가 다를 수 있으니 안전하게 정렬 또는 merge
    lgbm = lgbm.sort_values("game_episode").reset_index(drop=True)
    lstm = lstm.sort_values("game_episode").reset_index(drop=True)
    
    # 검증: game_episode가 일치하는지 확인
    if not np.array_equal(lgbm["game_episode"].values, lstm["game_episode"].values):
        print("Error: Game Episode IDs do not match!")
        return

    # -----------------------------------------------
    # [핵심] 가중치 설정 (Weighted Blending)
    # 성능이 좋은 LGBM에 더 많은 신뢰를 줍니다.
    # -----------------------------------------------
    w_lgbm = 0.7  # LGBM (14.48)
    w_lstm = 0.3  # LSTM (15.76)
    
    print(f"Blending... (LGBM: {w_lgbm} + LSTM: {w_lstm})")
    
    final_sub = lgbm.copy()
    final_sub["end_x"] = (lgbm["end_x"] * w_lgbm) + (lstm["end_x"] * w_lstm)
    final_sub["end_y"] = (lgbm["end_y"] * w_lgbm) + (lstm["end_y"] * w_lstm)
    
    final_sub.to_csv(save_path, index=False)
    print(f"Ensemble submission saved to: {save_path}")

if __name__ == "__main__":
    main()