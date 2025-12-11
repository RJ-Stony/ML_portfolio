import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import event2idx, result2idx

class HybridFinalPassLSTM(nn.Module):
    """
    [하이브리드 딥러닝 모델]
    - 시간 순서 데이터(LSTM)와 통계적 요약 정보(Linear)를 결합하여 예측 성능을 높임
    
    1. Bi-LSTM: 과거와 미래 방향을 모두 살펴서 전체적인 맥락 파악
    2. Last-K MLP: 골이 터지기 직전(마지막 K개) 상황을 집중 분석
    3. Global Features: 공격 스타일(클러스터 ID) 같은 거시적 정보 반영
    """
    def __init__(self, 
                 num_feats=12, 
                 event_emb_dim=6, 
                 result_emb_dim=3,
                 cluster_emb_dim=4, 
                 epi_feat_dim=7, 
                 hidden_dim=128, 
                 k_last=3, 
                 num_layers=1):
        super().__init__()
        
        # 1. 임베딩 레이어 (Embeddings)
        # 'Pass', 'Shot' 같은 문자열 카테고리를 숫자로 된 벡터(Vector)로 변환
        # -> 모델이 단어의 의미를 학습할 수 있게 됨
        self.event_emb = nn.Embedding(len(event2idx), event_emb_dim)
        self.result_emb = nn.Embedding(len(result2idx), result_emb_dim)
        self.cluster_emb = nn.Embedding(5, cluster_emb_dim) # 5가지 공격 스타일
        
        # LSTM에 들어갈 입력 크기 = 수치형 피처 + 이벤트 벡터 + 결과 벡터
        input_dim = num_feats + event_emb_dim + result_emb_dim

        # 2. Bi-LSTM (양방향 LSTM)
        # 데이터를 정방향, 역방향으로 모두 읽어서 앞뒤 문맥을 완벽히 파악함
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=True) # 양방향 True
        
        self.k_last = k_last
        
        # 3. Last-K MLP
        # 시퀀스 전체보다는 '마지막 순간'이 패스 성공에 더 중요할 수 있음
        # 그래서 마지막 K개의 정보만 따로 빼서 MLP(신경망)로 한 번 더 가공함
        # *2인 이유: Bi-LSTM이라서 순방향+역방향 히든 스테이트가 합쳐져 있음
        self.lastk_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 * k_last, hidden_dim),
            nn.ReLU(),       # 활성화 함수 (비선형성 추가)
            nn.Dropout(0.2)  # 과적합 방지 (일부 뉴런 끄기)
        )
        
        # 4. 최종 퓨전 레이어 (Final Fusion)
        # [전체 문맥] + [마지막 순간 정보] + [공격 스타일] + [통계 정보]를 모두 합침
        fusion_dim = (hidden_dim * 2) + hidden_dim + cluster_emb_dim + epi_feat_dim
        
        # 최종적으로 X, Y 좌표 2개를 예측
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128), # 데이터 분포를 정돈하여 학습 안정화
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2) # 출력: end_x, end_y
        )

    def forward(self, seq, ev, rs, lengths, cluster_id, epi_feats):
        # 1. 시퀀스 데이터 준비
        ev_e = self.event_emb(ev) # 이벤트 ID -> 벡터
        rs_e = self.result_emb(rs) # 결과 ID -> 벡터
        x = torch.cat([seq, ev_e, rs_e], dim=-1) # 모든 정보를 이어 붙임

        # 패킹(Packing): 패딩된 0 부분은 연산하지 않도록 압축 (속도 효율)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True) # 다시 압축 해제
        
        # 2. 전체 문맥 추출 (Global Context)
        # LSTM의 맨 마지막 상태(Hidden State)를 가져옴
        # Bi-LSTM이므로 순방향 마지막(h_n[-2]) + 역방향 마지막(h_n[-1]) 결합
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_context = torch.cat([h_fwd, h_bwd], dim=-1) 
        
        # 3. 세부 문맥 추출 (Local Context)
        # 시퀀스의 끝부분 K개를 슬라이싱해서 가져옴
        B, T, H2 = out_padded.size()
        lastk_list = []

        for i in range(B):
            L = lengths[i].item() # 실제 길이
            start = max(0, L - self.k_last) # 끝에서 K번째 위치
            lastk = out_padded[i, start:L]
            
            # 혹시 길이가 K보다 짧으면 0으로 채움
            if lastk.size(0) < self.k_last:
                pad = torch.zeros(self.k_last - lastk.size(0), H2, device=seq.device)
                lastk = torch.cat([pad, lastk], dim=0)
            
            lastk_list.append(lastk.reshape(-1)) # 일렬로 펼침

        lastk_tensor = torch.stack(lastk_list)
        h_lastk = self.lastk_mlp(lastk_tensor) # MLP 통과
        
        # 4. 하이브리드 특징 준비
        c_emb = self.cluster_emb(cluster_id) # 클러스터 ID -> 벡터
        
        # 5. 모든 정보 융합 및 예측 (Fusion)
        # [전체] + [최근] + [스타일] + [통계]
        combined = torch.cat([h_context, h_lastk, c_emb, epi_feats], dim=1)
        out = self.head(combined) # 최종 좌표 예측
        
        return out