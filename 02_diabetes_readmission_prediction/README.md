# 당뇨 환자 재입원 예측 모델링

## 1. 사전 세팅

---

## 2. 문제 정의

* 당뇨는 전 세계 성인의 9% 이상에 영향을 미치며 증가 추세
* **30일 이내 재입원 환자**를 예측하는 모델을 만들고, **에러 분석으로 성능 향상**을 도모

---

## 3. 데이터 확인

### 3.1. 데이터 명세

| Column   | Description                |
|:---------|:---------------------------|
| Features | 환자 입원 기간 내 관련 정보 	   |
| Class    | readmit_30_days            |

---

## 4. 문제 해결 프로세스 정의

* **문제**: 퇴원 환자 중 재입원하는 중증 환자 증가
* **기대 효과**: 재입원 환자 예측 → 사전 조치·모니터링, 생존율 향상
* **해결 방안**: 30일 이내 재입원 이진분류, Class Weight 영향·Feature Importance 분석
* **성과 측정**: 에러/Cohort 분석 기반 점진 개선

---

## 5. 데이터 전처리 및 EDA

### 5.1. 데이터 전처리

#### 5.1.1. Target 유의 변수 제거(누설 방지)

- `readmitted`, `readmit_binary` 분포 확인(타깃과 직접 상관 → **Feature로 사용 불가**)
- 타깃과 직접적 상관 변수를 포함하면 **과도한 정확도**를 보여 정상 모델로 볼 수 없음(데이터 누설)

#### 5.1.2. 인코딩

---

## 6. 모델링

### 6.1. LightGBM Classifier

- Class Weight, Feature Importance 확인
- Optuna로 하이퍼파라미터 자동 탐색
- AUC / Recall / Precision 등 모델 평가

---

## 7. 에러 분석

- 에러 분포 시각화 및 Cohort별 오탐 분석

---

## 8. Shapley value

- **Shapley value**를 활용해 변수 중요도 해석해 각 Feature가 모델 예측값에 미치는 영향을 시각화
- 재입원 확률에 가장 큰 영향을 준 변수들을 도출함
- 상위 Feature: `had_inpatient_days`, `discharge_disposition_id_'Discharged to Home'` 등

---

## 9. 후처리(Post Processing) 및 결론

- Positive 클래스 예측 시 임계값(Threshold) 조정 후 모델 재평가
- 확률 분포를 기반으로 재입원 위험군 재분류
- 초기 Baseline은 재입원 환자 탐지가 미흡했기에, Optuna 기반 하이퍼파라미터 튜닝으로 LightGBM 모델 성능(AUC·Recall)을 최적화함
- 클래스 불균형에 대응하여 True Positive 비율 확보와 False Negative 최소화를 목표로 학습 진행