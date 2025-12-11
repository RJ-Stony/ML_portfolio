def simplify_event(t: str) -> str:
    if t in ["Pass", "Pass_Freekick", "Pass_Corner"]:
        return "Pass"
    if t == "Carry":
        return "Carry"
    if t in ["Duel", "Tackle", "Interception", "Recovery"]:
        return "Duel_Turnover"
    if t == "Cross":
        return "Cross"
    if t.startswith("Shot") or t == "Penalty Kick":
        return "Shot"
    if t in ["Clearance", "Aerial Clearance"]:
        return "Clearance"
    if t in ["Catch", "Parry", "Goal Kick", "Keeper Rush-Out"]:
        return "GK_Action"
    if t in ["Block", "Deflection", "Intervention", "Hit"]:
        return "Deflect_Block"
    if t == "Throw-In":
        return "SetPiece"
    if t in ["Goal", "Own Goal"]:
        return "Goal_Event"
    if t in ["Error", "Out", "Foul", "Foul_Throw", "Handball_Foul", "Offside"]:
        return "Error_Out"
    return "Misc"

def simplify_result(result_name):
    if result_name in ["Successful", "On Target", "Goal"]:
        return "Success"
    if result_name in ["Unsuccessful", "Off Target", "Blocked"]:
        return "Fail"
    return "None"

EVENT_VOCAB = ['Carry', 'Clearance', 'Cross', 'Deflect_Block', 'Duel_Turnover', 'Error_Out', 'GK_Action', 'Misc', 'Pass', 'SetPiece', 'Shot']
RESULT_VOCAB = ['Fail', 'None', 'Success']

event2idx = {ev: i for i, ev in enumerate(EVENT_VOCAB)}
result2idx = {rs: i for i, rs in enumerate(RESULT_VOCAB)}