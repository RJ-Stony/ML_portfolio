import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import config
import feature_engineering

def make_last_k_data(df, K=20):
    # 1. 정렬 및 인덱스
    df = df.sort_values(["game_episode", "time_seconds", "action_id"]).reset_index(drop=True)
    df["event_idx"] = df.groupby("game_episode").cumcount()
    df["rev_idx"] = df.groupby("game_episode")["event_idx"].transform(lambda x: x.max() - x)
    
    # 2. FE
    df["dt"] = df.groupby("game_episode")["time_seconds"].diff().fillna(0)
    df["dx"] = df["end_x"] - df["start_x"]
    df["dy"] = df["end_y"] - df["start_y"]
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df["angle"] = np.arctan2(df["dy"], df["dx"])
    
    # [Leakage 방지] 마지막 이벤트 정보 마스킹
    mask_last = df["rev_idx"] == 0
    leak_cols = ["dx", "dy", "dist", "angle"]
    df.loc[mask_last, leak_cols] = 0
    
    # 3. Last K
    last_k_df = df[df["rev_idx"] < K].copy()
    last_k_df["pos"] = last_k_df["rev_idx"] 
    
    # 4. Cols
    num_cols = ["start_x", "start_y", "dx", "dy", "dist", "angle", "dt"]
    cat_cols = ["type_name", "result_name"]
    
    # 5. Pivot
    pivot_dfs = []
    for col in num_cols + cat_cols:
        p = last_k_df.pivot(index="game_episode", columns="pos", values=col)
        p.columns = [f"{col}_{int(c)}" for c in p.columns]
        pivot_dfs.append(p)
    wide_df = pd.concat(pivot_dfs, axis=1).reset_index()
    
    # 6. Meta & Target
    last_events = df.groupby("game_episode").tail(1).reset_index(drop=True)
    target = last_events[["game_episode", "end_x", "end_y"]].rename(columns={"end_x": "target_x", "end_y": "target_y"})
    meta = last_events[["game_episode", "game_id", "is_home"]]
    
    final_df = wide_df.merge(meta, on="game_episode", how="left")
    final_df = final_df.merge(target, on="game_episode", how="left")
    
    return final_df, num_cols, cat_cols

def main():
    print("Start LGBM (Target: Minimize Euclidean Distance via RMSE)...")
    
    # 1. Load Data
    print("Loading Data...")
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_test = feature_engineering.load_test_data(config.TEST_PATH)
    
    df_train["is_train"] = 1
    df_test["is_train"] = 0
    all_events = pd.concat([df_train, df_test], ignore_index=True)
    
    # 2. Make Features
    K = 20
    print(f"Building Last-{K} Features...")
    wide_df, num_feat, cat_feat = make_last_k_data(all_events, K=K)
    
    # 3. Label Encoding
    for col_base in cat_feat:
        for k in range(K):
            col_name = f"{col_base}_{k}"
            if col_name in wide_df.columns:
                le = LabelEncoder()
                wide_df[col_name] = wide_df[col_name].astype(str)
                wide_df[col_name] = le.fit_transform(wide_df[col_name])
    
    # 4. Split
    train_df = wide_df[wide_df["game_episode"].isin(df_train["game_episode"])].copy()
    test_df = wide_df[wide_df["game_episode"].isin(df_test["game_episode"])].copy()
    
    drop_cols = ["game_episode", "game_id", "target_x", "target_y", "is_train"]
    features = [c for c in train_df.columns if c not in drop_cols]
    
    # 5. Train
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df["game_id"]
    
    y_cols = ["target_x", "target_y"]
    models = {col: [] for col in y_cols}
    oof_preds = {col: np.zeros(len(train_df)) for col in y_cols}
    
    # 하이퍼파라미터 (정밀 학습을 위해 learning_rate 낮춤)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,  # 0.03 -> 0.01 (더 천천히, 꼼꼼하게)
        "n_estimators": 10000,  # 에포크 늘림
        "max_depth": 8,
        "num_leaves": 63,
        "colsample_bytree": 0.7,
        "subsample": 0.7,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }
    
    for y_col in y_cols:
        print(f"\n================ Training Model for {y_col} ================")
        y = train_df[y_col]
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, y, groups)):
            X_train, y_train = train_df.iloc[train_idx][features], y.iloc[train_idx]
            X_val, y_val = train_df.iloc[val_idx][features], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**lgb_params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200), # 참을성 증가
                    lgb.log_evaluation(period=1000) # 로그 덜 시끄럽게
                ]
            )
            
            models[y_col].append(model)
            oof_preds[y_col][val_idx] = model.predict(X_val)
            
    # 6. 진짜 점수 확인 (Euclidean Distance)
    print("\n" + "="*50)
    print("FINAL VALIDATION SCORE (Euclidean Distance)")
    print("="*50)
    
    dx = oof_preds["target_x"] - train_df["target_x"]
    dy = oof_preds["target_y"] - train_df["target_y"]
    oof_dist = np.sqrt(dx**2 + dy**2).mean()
    
    print(f"\nOOF Mean Euclidean Distance: {oof_dist:.4f} m")
    print("\n(이 점수가 실제 리더보드 점수와 가장 비슷합니다!)")
    print("="*50)
    
    # 7. Submit
    print("\nPredicting Test...")
    sub_x = np.zeros(len(test_df))
    sub_y = np.zeros(len(test_df))
    
    for model in models["target_x"]:
        sub_x += model.predict(test_df[features]) / n_splits
    for model in models["target_y"]:
        sub_y += model.predict(test_df[features]) / n_splits
        
    submission = pd.read_csv(config.SUBMISSION_PATH)
    pred_df = pd.DataFrame({"game_episode": test_df["game_episode"], "end_x": sub_x, "end_y": sub_y})
    
    if "end_x" in submission.columns:
        submission = submission.drop(columns=["end_x", "end_y"])
    
    final_sub = submission.merge(pred_df, on="game_episode", how="left").fillna(0)
    final_sub["end_x"] = final_sub["end_x"].clip(0, 105)
    final_sub["end_y"] = final_sub["end_y"].clip(0, 68)
    
    final_sub.to_csv("../Data/lgbm_submission.csv", index=False)
    print("Saved: ../Data/lgbm_submission.csv")

if __name__ == "__main__":
    main()