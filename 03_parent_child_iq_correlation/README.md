# 부모와 자녀의 IQ 상관 분석

## 1. 사전 세팅

---

## 2. 문제 정의

* 부모의 IQ가 자녀의 IQ에 어떤 영향을 미치는지 분석
* 단순 상관관계를 넘어 회귀 분석을 통해 설명력 평가
* 대신 오래된 데이터이기 때문에 현대 정서를 반영하기 힘들기에 단순하게만 분석

---

## 3. 데이터 확인

### 3.1. 데이터 명세

| Column    | Description                 |
| :-------- | :-------------------------- |
| kid_score | 자녀의 IQ 점수                   |
| mom_hs    | 어머니의 고등학교 졸업 여부           |
| mom_iq_c(추가 컬럼)  | 어머니의 IQ (centered variable) |

---

## 4. 문제 해결 프로세스 정의

* 문제: 아이의 IQ와 엄마의 IQ의 실제 상관 여부 검증
* 해결 방안: OLS 회귀모델 및 K-Fold Cross Validation 적용
* 성과 측정: R² 및 RMSE 기반 모델 설명력 확인

---

## 5. 데이터 전처리 및 EDA

* OLS(최소제곱법) 회귀 적용
* 독립변수: mom_hs, mom_iq_c, mom_age_c
* 회귀 결과: R² = 0.215, RMSE ≈ 18.06
* A/B Line 시각화: mom_hs 여부에 따른 회귀선 비교

---

## 6. 교차검증(Cross Validation)

* KFold(5분할) 로 R², RMSE 검증
* 직접 구현 및 sklearn 비교를 통해 일관된 결과 확인

### 6.3. Interaction term 추가

* 상호작용 변수 생성:
  `df['mom_hs_iq_c'] = df['mom_hs'] * df['mom_iq_c']`
* 회귀식 확장:
  `kid_score = β0 + β1*mom_hs + β2*mom_iq_c + β3*(mom_hs*mom_iq_c) + ε`
* 5-Fold CV를 통해 모델 성능(R², RMSE) 재측정

---

## 7. 결론

* 상호작용항 추가 시 모델 설명력 향상
* 교차검증 결과, 모델의 일반화 성능(R² 및 RMSE)이 안정적으로 유지
* CV는 데이터 분포 확보 및 복잡한 모델 검증 시 유용