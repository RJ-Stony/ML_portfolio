"""
- 데이터 전처리 과정에서 이벤트 이름이나 결과(성공/실패 등)를 단순화해,
- 모델이 학습하기 쉽도록 만드는 보조 함수들 정의
- 문자열로 된 이벤트/결과를 숫자로 바꾸기 위한 사전도 정의
"""

def simplify_event(t: str) -> str:
    """
    다양한 이벤트 타입(type_name)을 더 적은 종류의 카테고리로 단순화
    예: 'Pass_Freekick', 'Pass_Corner' -> 모두 'Pass'로 통일
    
    Args:
        t (str): 원본 이벤트 타입 이름 (예: "Pass_Freekick")
    
    Returns:
        str: 단순화된 이벤트 이름 (예: "Pass")
    """
    # 패스 관련 이벤트들을 하나로 묶음
    if t in ["Pass", "Pass_Freekick", "Pass_Corner"]:
        return "Pass"
    
    # 드리블은 그대로 유지
    if t == "Carry":
        return "Carry"
    
    # 수비 경합, 태클, 가로채기 등 소유권 다툼 관련 이벤트를 하나로 묶음
    if t in ["Duel", "Tackle", "Interception", "Recovery"]:
        return "Duel_Turnover"  # 턴오버(공수교대)가 일어날 수 있는 상황들
    
    # 크로스는 패스와 구별하여 별도로 유지 (공격적 중요도가 다를 수 있음)
    if t == "Cross":
        return "Cross"
    
    # 슈팅, 페널티킥은 모두 'Shot'으로 통일
    if t.startswith("Shot") or t == "Penalty Kick":
        return "Shot"
    
    # 걷어내기 관련 이벤트 통일
    if t in ["Clearance", "Aerial Clearance"]:
        return "Clearance"
    
    # 골키퍼의 방어 행동 등 통일
    if t in ["Catch", "Parry", "Goal Kick", "Keeper Rush-Out"]:
        return "GK_Action"
    
    # 슛을 막거나 굴절되는 상황들 통일
    if t in ["Block", "Deflection", "Intervention", "Hit"]:
        return "Deflect_Block"
    
    # 스로인은 세트피스(SetPiece)로 분류
    if t == "Throw-In":
        return "SetPiece"
    
    # 골과 자책골은 득점 상황(Goal_Event)으로 분류
    if t in ["Goal", "Own Goal"]:
        return "Goal_Event"
    
    # 파울, 오프사이드, 공이 나감 등 경기가 중단되거나 흐름이 끊기는 상황들
    if t in ["Error", "Out", "Foul", "Foul_Throw", "Handball_Foul", "Offside"]:
        return "Error_Out"
    
    # 위 조건에 해당하지 않는 나머지는 'Misc'(기타)로 분류
    return "Misc"

def simplify_result(result_name):
    """
    이벤트의 결과(성공, 실패 등)를 3가지(성공, 실패, 없음)로 단순화
    
    Args:
        result_name (str): 원본 결과 이름 (예: "On Target", "Off Target")
        
    Returns:
        str: "Success", "Fail", 또는 "None"
    """
    # 긍정적인 결과들 -> Success
    if result_name in ["Successful", "On Target", "Goal"]:
        return "Success"
    
    # 부정적인 결과들 -> Fail
    if result_name in ["Unsuccessful", "Off Target", "Blocked"]:
        return "Fail"
    
    # 결과가 명시되지 않거나 판단이 모호한 경우 -> None
    return "None"

# 모델이 학습할 때 사용할 이벤트 어휘 사전 (가나다 순 정렬)
EVENT_VOCAB = ['Carry', 'Clearance', 'Cross', 'Deflect_Block', 'Duel_Turnover', 'Error_Out', 'GK_Action', 'Misc', 'Pass', 'SetPiece', 'Shot']

# 모델이 학습할 때 사용할 결과 어휘 사전
RESULT_VOCAB = ['Fail', 'None', 'Success']

# 문자열 이벤트를 숫자 인덱스로 변환하기 위한 맵핑 (예: 'Pass' -> 8)
event2idx = {ev: i for i, ev in enumerate(EVENT_VOCAB)}

# 문자열 결과를 숫자 인덱스로 변환하기 위한 맵핑 (예: 'Success' -> 2)
result2idx = {rs: i for i, rs in enumerate(RESULT_VOCAB)}