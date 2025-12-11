import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

def process_dataframe(df):
    """
    [데이터 요약/가공 함수]
    - 원본 이벤트 로그를 받아서 '에피소드(공격 시도)' 단위의 요약 특징 생성
    - 예: 패스를 얼마나 많이 했는지, 총 이동 거리는 얼마인지 등
    """
    episode_features = []
    
    # 경기 에피소드(game_episode)별로 데이터를 나눔
    for ge, g in tqdm(df.groupby("game_episode"), desc="Extracting Features"):
        # 시간 순서대로 정렬 (안 되어 있을 경우를 대비)
        g = g.sort_values(["time_seconds", "action_id"])
        
        # 이동 거리 및 각도 계산
        # diff(): 이전 행과의 차이를 구함
        dx = g["start_x"].diff().fillna(0).values
        dy = g["start_y"].diff().fillna(0).values
        angle = np.arctan2(dy, dx)
        
        # 해당 에피소드의 특징을 요약해서 사전(Dictionary)으로 만듦
        feat = {
            "game_episode": ge,
            "epi_len": len(g), # 이벤트 개수 (공격 지속 시간 간접 파악)
            "ratio_pass": (g["type_name"] == "Pass").mean(), # 패스 비중 (티키타카 여부?)
            "ratio_carry": (g["type_name"] == "Carry").mean(), # 드리블 비중
            "cum_dx": dx.sum(), # X축 총 이동량
            "cum_dy": dy.sum(), # Y축 총 이동량
            "movement_norm": np.sqrt(dx.sum()**2 + dy.sum()**2), # 직선 이동 거리
            "angle_std": np.std(angle) if len(angle) > 0 else 0, # 방향 변화의 다양성 (속임수 동작?)
        }
        episode_features.append(feat)
        
    return pd.DataFrame(episode_features)

def load_test_data(test_meta_path):
    """
    [테스트 데이터 로드 함수]
    - test.csv(파일 경로 목록)를 읽어서, 흩어져 있는 개별 csv 파일들을 로딩
    - 하나로 합쳐서 거대한 DataFrame 반환
    """
    test_meta = pd.read_csv(test_meta_path)
    base_dir = os.path.dirname(test_meta_path) # 데이터 폴더 위치 파악
    
    all_dfs = []
    print(f"Loading Test Files from {base_dir}...")
    
    for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="Loading Test"):
        # 파일 경로의 불필요한 부분(./) 정리
        clean_path = row["path"].replace("./", "")
        full_path = os.path.join(base_dir, clean_path)
        
        try:
            df = pd.read_csv(full_path)
            # 중요: 개별 파일에는 '어떤 에피소드인지' 정보가 없을 수 있음
            # 메타데이터에 있는 game_episode ID를 강제로 넣어줌
            df["game_episode"] = row["game_episode"]
            all_dfs.append(df)
        except FileNotFoundError:
            # 파일이 실수로 누락된 경우, 프로그램이 죽지 않고 넘어가도록 처리
            continue
            
    if not all_dfs:
        raise ValueError("No test files loaded! Check your data directory structure.")
        
    return pd.concat(all_dfs, ignore_index=True)

def generate_hybrid_features(train_path, test_path, save_path="Data/"):
    """
    [하이브리드 피처 생성 메인 함수]
    - 수치형 요약 피처를 만들고, K-Means로 공격 스타일을 그룹화(Clustering)함
    - 만들어진 피처와 클러스터 정보를 파일로 저장
    """
    # 1. Train 데이터 로드 및 피처 생성
    print(f"Loading Train Data: {train_path}")
    df_train = pd.read_csv(train_path)
    train_epi_df = process_dataframe(df_train)
    
    # 2. Test 데이터 로드 및 피처 생성
    # Test 데이터는 여러 파일로 쪼개져 있어서 합치는 과정 필요
    print(f"Loading Test Data: {test_path}")
    df_test = load_test_data(test_path)
    test_epi_df = process_dataframe(df_test)
    
    # 3. K-Means Clustering 수행
    # 공격 패턴이 비슷한 것끼리 묶어서 '공격 스타일ID'를 만듦 -> 모델에게 힌트 제공
    print("Fitting Clustering Model...")
    cluster_cols = ["epi_len", "ratio_pass", "ratio_carry", "cum_dx", "cum_dy", "angle_std"]
    
    # 혹시 모를 결측치(NaN)를 0으로 채움
    train_epi_df[cluster_cols] = train_epi_df[cluster_cols].fillna(0)
    test_epi_df[cluster_cols] = test_epi_df[cluster_cols].fillna(0)
    
    # 데이터 스케일링 (값의 범위를 맞춰줌, 예: 패스비율 0~1 vs 이동거리 0~100)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_epi_df[cluster_cols])
    X_test = scaler.transform(test_epi_df[cluster_cols])
    
    # 5개의 그룹으로 나누기
    kmeans = KMeans(n_clusters=5, random_state=42)
    
    # Train 데이터로 패턴을 학습하고, Train/Test 각각 어떤 그룹인지 예측
    train_epi_df["cluster_id"] = kmeans.fit_predict(X_train)
    test_epi_df["cluster_id"] = kmeans.predict(X_test)
    
    # 4. 결과 저장
    # 스케일러와 모델도 나중에 쓸 수 있으니 저장 (Pickle)
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_path, "kmeans.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
        
    train_save_name = os.path.join(save_path, "train_episode_features.csv")
    test_save_name = os.path.join(save_path, "test_episode_features.csv")
    
    train_epi_df.to_csv(train_save_name, index=False)
    test_epi_df.to_csv(test_save_name, index=False)
    
    print(f"Features saved to:\n - {train_save_name}\n - {test_save_name}")

if __name__ == "__main__":
    # 파일을 직접 실행했을 때만 동작하는 테스트 코드
    generate_hybrid_features("../Data/train.csv", "../Data/test.csv", "../Data/")