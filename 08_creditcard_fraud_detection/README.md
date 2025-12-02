# 신용카드 이상거래 탐지 모델링

---

## 1. 사전 세팅

---

## 2. 문제 정의
- 신용카드 거래 중 극히 일부만 발생하는 이상거래(Fraud)를 조기에 탐지하는 자동 분류 모델 구축  
- 비식별화된 거래 데이터를 활용해 정상·이상 패턴을 학습하고 고객 피해를 최소화하는 탐지 체계 수립

---

## 3. 데이터 간단 확인
| Column | Description |
|--------|-------------|
| class | 이상거래 여부 |
| v1 ~ v28 | PCA 기반 비식별 Feature |
| log_amount | 거래 금액 로그 변환 값 |

- 총 **284,807행 × 30열**
- 정상 284,315건(99.83%) / 이상거래 492건(0.17%)
- 결측치 없음  
- 모든 Feature는 비식별화되어 있어 구체적 의미 파악 불가

---

## 4. 문제 해결 프로세스 정의
**문제:** 극단적 불균형(0.17%)으로 인한 Fraud 탐지 어려움  
**기대 효과:** 조기 탐지로 고객 피해 최소화 및 FDS(Fraud Detection System) 보조  
**해결 방안:**  
- Under-sampling 기반 데이터 재구성  
- Logistic, KNN, LDA, SVM, RandomForest 모델 비교  
- Fraud Recall 중심 성능 평가  
**성과 측정:** Accuracy, Precision, Recall, F1-score, ROC-AUC  
**현업 적용:** 경보 시스템의 추가 탐지 로직으로 활용 가능

---

## 5. 데이터 확인
- 정상/이상 거래 클래스 비율 심각한 불균형 → 99.83% vs. 0.17%
- 정상·이상 거래에서 각각 **5%만 랜덤 추출**해 학습 가능한 규모로 조정  
- 최종 Under-sampled 데이터: **14,241건 (정상 14,216 / Fraud 25)**
- Feature 분포 분석(distplot), 상관 히트맵, Pairplot 수행
- 수치형 변수에 StandardScaler 적용
- Stratified Split으로 학습/평가 데이터 간 클래스 비율 유지

---

## 6. Data Readiness Check
- 표준화된 상태에서 Fraud/Not Fraud 분포 구조 시각화  
- Feature 간 상관 분석 및 결정 경계 파악
- Fraud 데이터 희소성으로 패턴 확인에 한계 존재

---

## 7. 모델링 비교
| Framework | Model | Accuracy | Recall(Fraud) | 특징 |
|-----------|--------|----------|----------------|--------|
| sklearn | Logistic RegressionCV | 1.00 | 0.62 | 안정적이나 Fraud 탐지 한계 |
| sklearn | KNN | 1.00 | 0.62 | 거리 기반, 희귀 패턴 약함 |
| sklearn | **LDA** | 1.00 | **0.75** | 가장 안정적 Fraud 탐지 |
| sklearn | SVM (Sigmoid Kernel) | 1.00 | 0.50 | 커널 부적합, 비선형 확장 부족 |
| sklearn | RandomForest | 1.00 | 0.62 | Permutation Importance 분석용 |

---

## 8. Feature Engineering: Over-sampling
- Over-sampling(Numpy & imblearn)도 실험  
- Over-sampling(1:1)은 분포 왜곡 → 오탐 증가 및 F1-score 저하 확인

---

## 9. 모델 비교
- RF 기반 Permutation Importance: v1·v4·v5·v28 등이 상대적으로 높은 기여
- Fraud 데이터 희소성 → RandomForest 해석력 제한
- 교차검증(5-Fold)에서 LDA가 평균 F1-score 기준 가장 안정적 성능 확인

---

## 10. 결론
- 극단적 불균형 데이터에서는 **Over-sampling보다 Under-sampling이 실제 탐지력 향상에 더 효과적**
- Over-sampling(1:1)은 분포 왜곡 및 오탐 증가 → F1-score 대폭 저하  
- Fraud 도메인 특성상 Recall 확보가 핵심 → **LDA가 최종 후보 모델로 선정**
- Sigmoid 커널 기반 SVM은 데이터 구조 확장 부족으로 부적합  
- 향후 SHAP, Autoencoder 기반 이상탐지 등 확장 가능성 존재
