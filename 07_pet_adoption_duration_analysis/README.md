# 유기 동물 입양 소요 기간 분석

## 1. 사전 세팅

---

## 2. 문제 정의

* 유기된 애완동물 중 **어떤 특성을 가진 개체가 더 빨리 입양되는지** 분석
* 통제 불가능 변수(나이, 종, 색)와 통제 가능한 변수(사진 수, 설명 길이)를 구분

---

## 3. 데이터 간단 확인

| Column                  | Description     |
| :---------------------- | :-------------- |
| Type                    | 동물 종류 (개/고양이)   |
| Age                     | 나이              |
| Breed1                  | 종               |
| Color1 / Color2         | 주요 색상           |
| MaturitySize            | 크기              |
| FurLength               | 털 길이            |
| Vaccinated / Sterilized | 백신, 중성화 여부      |
| Health                  | 건강 상태           |
| Fee                     | 분양 비용           |
| Description             | 동물 소개 글         |
| PhotoAmt                | 등록 사진 수         |
| AdoptionSpeed           | 입양 속도 (작을수록 빠름) |

* 총 11,537행 × 15열
* 결측치 9건 (Description) 제거 후 11,528행 사용

---

## 4. 문제 해결 프로세스 정의

* **문제:** 어떤 요인이 입양 속도에 가장 큰 영향을 주는가
* **기대 효과:** 사진 수·설명 길이 등 통제 가능한 요소 최적화
* **해결 방안:**
  * 통계적 접근(Logistic Regression, Marginal Effect)
  * 머신러닝(RandomForest)
  * 딥러닝(TensorFlow NN) 비교
* **성과 측정:** Accuracy, Precision, Recall
* **현업 적용:** 보호소/입양 플랫폼의 프로필 작성 가이드라인 수립

---

## 5. 데이터 확인

---

## 6. Feature Engineering

* `Description` 결측 제거, `Adopted` (입양 여부) 생성
* 설명 길이, 단어 수, 평균 단어 길이 등 텍스트 기반 변수 추가
* 범주형 변수 원-핫 인코딩 (색상, 크기, 종 등)
* Age는 5분위로 나눠 구간화
* 모든 수치형 변수는 SMD(Standardized Mean Difference)로 표준화

---

## 7. Data Readiness Check

* 사진 수(`PhotoAmt`)가 많을수록 입양 확률이 높음
* 설명 길이(`DescriptionLength`)는 1 표준단위 증가 시 입양 확률 +2%p
* 평균 단어 길이는 유의미한 영향 미비
* 시각화: 상관 히트맵, Boxplot, Quantile별 입양률 등

---

## 8. 모델링

| Framework      | 모델                  | Accuracy | Precision | Recall | 특징                    |
| :------------- | :------------------ | :------: | :-------: | :----: | :-------------------- |
| `statsmodels`  | Logistic Regression |   0.73   |    0.73   |  1.00  | 사진 수 영향 통계적 유의        |
| `scikit-learn` | Logistic Regression |   0.73   |    0.73   |  1.00  | 동일 결과 재현              |
| `scikit-learn` | RandomForest        |     -    |     -     |    -   | Feature Importance 분석 |
| `TensorFlow`   | Dense NN (2층)       |   0.72   |    0.79   |  0.74  | 다변량 관계 자동 학습          |

---

## 9. 시뮬레이션 및 모델 비교

| 변경 요인       | 입양 확률 변화 |
| :---------- | :------: |
| 사진 2장 → 50장 |  +20.4%p |
| 나이 3 → 1세   |  +6.6%p  |
| 설명 길이 ↑     |  +1~2%p  |

---

## 10. 결론

* **사진 수와 설명 길이**는 입양 확률에 직접적 영향
* Logistic 모델에서 사진 수 1단위↑ → 입양 확률 +5.6%p
* 설명 길이 1단위↑ → 입양 확률 +2%p
* 통제 가능한 변수 최적화로 입양률 향상 가능
* 텍스트 기반 변수의 추가 활용 여지 있음 (감정 분석 등)