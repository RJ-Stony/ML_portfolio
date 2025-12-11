import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import event2idx, result2idx

"""
- 시퀀스 데이터를 받아 최종 패스의 위치를 예측하는 딥러닝 모델 정의
- LSTM 기반으로 하며, 마지막 K개 이벤트의 정보를 활용
"""

class FinalPassLSTMWithLastK(nn.Module):
    """
    LSTM 사용하여 시계열 데이터 처리하고, 마지막 K개 은닉 상태(Hidden State) 사용해 최종 예측 수행
    """
    def __init__(self, num_feats=12, event_emb_dim=6, result_emb_dim=3, hidden_dim=96, k_last=3, num_layers=1):
        super().__init__()
        
        # 1. 임베딩 레이어 (Embedding Layer)
        # 범주형 데이터(이벤트 종류, 결과 종류)를 Dense Vector로 변환
        # 예: 'Pass' 단어 하나를 [0.1, -0.5, ...] 같은 특징 벡터로 바꾸는 과정
        self.event_emb = nn.Embedding(len(event2idx), event_emb_dim)
        self.result_emb = nn.Embedding(len(result2idx), result_emb_dim)
        
        # LSTM에 들어갈 입력 차원 계산 (수치형 피쳐 + 이벤트 임베딩 + 결과 임베딩)
        input_dim = num_feats + event_emb_dim + result_emb_dim

        # 2. LSTM 레이어
        # 시계열 데이터를 순차적으로 처리하여 각 시점의 은닉 상태(Hidden State) 계산
        # batch_first=True -> 입력/output의 첫 차원이 batch 차원이 되도록 함
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # 마지막 K개 스텝만 더 집중해서 볼 것
        self.k_last = k_last
        
        # 3. Last-K 처리용 MLP
        # 마지막 K개 히든 스테이트 일렬로 펼쳐서 정보 압축/가공
        self.lastk_mlp = nn.Sequential(
            nn.Linear(hidden_dim * k_last, hidden_dim),
            nn.ReLU(),
        )
        
        # 4. 최종 예측 레이어
        # LSTM 마지막 상태(맥락 정보) + Last-K 정보(최근 정보) 합쳐서
        # 최종적으로 x, y 좌표 2개 예측
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, seq, ev, rs, lengths):
        """
        모델의 Forward 과정
        
        Args:
            seq (Tensor): 수치형 피쳐 시퀀스 (Batch, Time, NumFeats)
            ev (Tensor): 이벤트 인덱스 시퀀스 (Batch, Time)
            rs (Tensor): 결과 인덱스 시퀀스 (Batch, Time)
            lengths (Tensor): 각 시퀀스 실제 길이 (Batch,)
        """
        # 1. 임베딩(Embedding) 적용
        ev_e = self.event_emb(ev) # (B, T, event_emb_dim)
        rs_e = self.result_emb(rs) # (B, T, result_emb_dim)
        
        # 2. 입력 벡터 생성 (수치형 + 임베딩)
        x = torch.cat([seq, ev_e, rs_e], dim=-1) # (B, T, InputDim)

        # 3. 패킹(Packing) - 가변 길이 시퀀스 효율적으로 처리
        # 패딩된 0 부분은 계산하지 않도록 해, 성능과 정확도 높임
        # cpu() -> GPU에서 실행할 때 메모리 효율화
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM 통과
        packed_out, (h_n, c_n) = self.lstm(packed)
        
        # 언패킹(Unpacking) - 다시 원래 (Batch, Time, Hidden) 형태로 복원
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)

        # 4. 전체 맥락 정보 (Context)
        # h_n[-1]은 LSTM이 시퀀스를 끝까지 읽고 난 후의 최종 요약 정보
        h_context = h_n[-1]
        
        # 5. 최근 K개 정보 (Recent Context) 추출
        # 시퀀스가 끝나기 직전 K개 스텝이 패스 성공 여부에 가장 중요할 것이라는 가설
        B, T, H = out_padded.size()
        lastk_list = []

        for i in range(B):
            L = lengths[i].item() # 실제 길이
            end = L
            start = max(0, end - self.k_last) # 끝에서 K개 전부터
            
            # 유효한 마지막 구간 추출
            lastk = out_padded[i, start:end]
            
            # 만약 시퀀스 길이가 K보다 짧으면, 부족한 부분은 0으로 채움 (Padding)
            if lastk.size(0) < self.k_last:
                pad = torch.zeros(self.k_last - lastk.size(0), H, device=seq.device)
                lastk = torch.cat([pad, lastk], dim=0)
            
            # (K, Hidden) -> (K * Hidden) 으로 펼침
            lastk_list.append(lastk.reshape(-1))

        lastk_tensor = torch.stack(lastk_list)
        
        # Last-K 정보를 MLP 통과시켜 요약
        h_lastk = self.lastk_mlp(lastk_tensor)
        
        # 6. 최종 결합 및 예측
        # 전체 맥락(h_context) + 최근 맥락(h_lastk) 결합
        h = torch.cat([h_context, h_lastk], dim=1)
        
        # (x, y) 예측값 출력
        out = self.fc(h)
        return out