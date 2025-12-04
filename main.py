import yaml
import requests
import urllib.parse
import itertools
import numpy as np
import os
import urllib3
from scipy.optimize import linear_sum_assignment
import unicodedata

def pad_text(text, width):
    current_width = 0
    for char in text:
        # 동아시아 문자(한글 등)는 너비를 2로 계산, 그 외는 1
        if unicodedata.east_asian_width(char) in ['F', 'W']:
            current_width += 2
        else:
            current_width += 1
            
    padding_size = max(0, width - current_width)
    return text + " " * padding_size

# ==========================================
# 1. 설정 및 초기화 (Configuration)
# ==========================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ENV_DIR = 'env'
FILE_KEY = os.path.join(ENV_DIR, 'key.yaml')
FILE_CLASS = os.path.join(ENV_DIR, 'classification.yaml')
FILE_DB = os.path.join(ENV_DIR, 'db.yaml')
FILE_INPUT = os.path.join(ENV_DIR, 'input_players.yaml')

# 하이퍼파라미터
ALPHA = 2.0       # 티어 점수 가중치
NOISE_STD = 0.1   # 노이즈 표준편차
MAX_TIER_SCORE = 3600.0

TIER_SCORES = {
    "I": 0, "B": 400, "S": 800, "G": 1200, "P": 1600, 
    "E": 2000, "D": 2400, "M": 2800, "GM": 3200, "C": 3600
}
DIVISIONS = {"4": 0, "3": 100, "2": 200, "1": 300}
ROLES = ["TOP", "JG", "MID", "ADC", "SUP"]

# ==========================================
# 2. 데이터 매니저 (Data Manager)
# ==========================================
class DataManager:
    def __init__(self):
        if not os.path.exists(FILE_KEY):
            raise FileNotFoundError(f"API Key file not found: {FILE_KEY}")
            
        with open(FILE_KEY, 'r') as f:
            self.api_key = yaml.safe_load(f)['API_KEY']
        
        self.headers = {
            "X-Riot-Token": self.api_key,
            "User-Agent": "ADC-Balancer/1.0"
        }
        
        self.host_v5 = "https://asia.api.riotgames.com"
        self.host_v4 = "https://kr.api.riotgames.com"
        
        if os.path.exists(FILE_DB):
            with open(FILE_DB, 'r', encoding='utf-8') as f:
                self.db = yaml.safe_load(f) or []
        else:
            self.db = []
            
    def save_db(self):
        with open(FILE_DB, 'w', encoding='utf-8') as f:
            yaml.dump(self.db, f, allow_unicode=True)

    def find_in_db(self, name, tag):
        for p in self.db:
            if p['name'] == name and p['tag'] == tag:
                return p
        return None

    def _fallback_get_id(self, puuid):
        print("   -> [WARN] Summoner-V4 failed. Trying Match-V5 fallback...")
        try:
            url = f"{self.host_v5}/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=1"
            res = requests.get(url, headers=self.headers, verify=False)
            if not res.ok or not res.json(): return None
            
            match_id = res.json()[0]
            url_detail = f"{self.host_v5}/lol/match/v5/matches/{match_id}"
            res_detail = requests.get(url_detail, headers=self.headers, verify=False)
            if not res_detail.ok: return None
            
            parts = res_detail.json()['info']['participants']
            for p in parts:
                if p['puuid'] == puuid:
                    print(f"   -> [INFO] Found ID from match history.")
                    return p['summonerId']
            return None
        except Exception as e:
            print(f"   -> [ERROR] Fallback Failed: {e}")
            return None

    def fetch_player_data(self, name, tag):
        print(f"[API] Fetching: {name}#{tag}")
        try:
            # Step 1. Account-V1
            s_name = urllib.parse.quote(name)
            s_tag = urllib.parse.quote(tag)
            url_acc = f"{self.host_v5}/riot/account/v1/accounts/by-riot-id/{s_name}/{s_tag}"
            
            res = requests.get(url_acc, headers=self.headers, verify=False)
            if res.status_code != 200:
                print(f"[ERROR] Account-V1 Error: {res.status_code} {res.text}")
                return None
            puuid = res.json().get('puuid')

            # Step 2. Summoner-V4
            url_sum = f"{self.host_v4}/lol/summoner/v4/summoners/by-puuid/{puuid}"
            res = requests.get(url_sum, headers=self.headers, verify=False)
            
            summ_id = None
            if res.status_code == 200:
                summ_data = res.json()
                summ_id = summ_data.get('id')
            
            if not summ_id:
                summ_id = self._fallback_get_id(puuid)
            
            if not summ_id:
                print("[ERROR] Critical: Failed to retrieve Summoner ID.")
                return None
            
            # Step 3. League-V4
            url_rank = f"{self.host_v4}/lol/league/v4/entries/by-summoner/{summ_id}"
            res = requests.get(url_rank, headers=self.headers, verify=False)
            rank_data = res.json()
            
            tier, division = "Unranked", "0"
            if isinstance(rank_data, list):
                for r in rank_data:
                    if r.get('queueType') == "RANKED_SOLO_5x5":
                        tier = r.get('tier', 'Unranked')[0]
                        if r.get('tier') == "GRANDMASTER": tier = "GM"
                        elif r.get('tier') == "CHALLENGER": tier = "C"
                        
                        div_map = {"I": "1", "II": "2", "III": "3", "IV": "4"}
                        division = div_map.get(r.get('rank'), "4")
                        break
            
            # Step 4. Mastery-V4
            url_mast = f"{self.host_v4}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count=100"
            res = requests.get(url_mast, headers=self.headers, verify=False)
            mastery = res.json()
            
            new_data = {
                "name": name, "tag": tag, "id": puuid,
                "tier": tier, "division": division,
                "mastery": mastery
            }
            
            existing = self.find_in_db(name, tag)
            if existing: self.db.remove(existing)
            self.db.append(new_data)
            self.save_db()
            return new_data
            
        except Exception as e:
            print(f"[ERROR] Exception for {name}#{tag}: {e}")
            return None

# ==========================================
# 3. ADC 모델 코어 (Model Core)
# ==========================================
class ADCModel:
    def __init__(self):
        if not os.path.exists(FILE_CLASS):
            print(f"[WARN] {FILE_CLASS} not found. Using empty classification.")
            self.champ_class = {}
        else:
            with open(FILE_CLASS, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.champ_class = data if data is not None else {}

    def calc_line_value(self, mastery_list):
        role_points = {r: 0.0 for r in ROLES}
        for m in mastery_list:
            cid = m['championId']
            points = m.get('championPoints', m.get('points', 0))
            
            possible_roles = self.champ_class.get(cid, []) 
            if not possible_roles: continue

            for role in possible_roles:
                role_points[role] += points
        
        total = sum(role_points.values())
        if total == 0: return {r: 0.2 for r in ROLES}
        return {r: v/total for r, v in role_points.items()}

    def calc_ability_value(self, tier, division):
        raw = TIER_SCORES.get(tier, 0) + DIVISIONS.get(str(division), 0)
        return raw / MAX_TIER_SCORE

    def calc_effective_power(self, p_data):
        L = self.calc_line_value(p_data.get('mastery', []))
        A = self.calc_ability_value(p_data.get('tier'), p_data.get('division'))
        epsilon = np.random.normal(0, NOISE_STD)
        
        S = {}
        for role in ROLES:
            S[role] = L[role] + (ALPHA * A) + epsilon
            
        return {"L": L, "A": A, "S": S, 
                "raw_tier": f"{p_data.get('tier')}{p_data.get('division')}"}

    def solve_team_assignment(self, team_players):
        cost_matrix = []
        for p in team_players:
            cost_matrix.append([-p['stats']['S'][r] for r in ROLES])
            
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        total_power = 0
        details = []
        for i in range(5):
            pid, rid = row_ind[i], col_ind[i]
            score = -cost_matrix[pid][rid]
            total_power += score
            details.append({
                "name": team_players[pid]['name'],
                "role": ROLES[rid],
                "score": score,
                "tier": team_players[pid]['stats']['raw_tier'],
                "L": team_players[pid]['stats']['L'][ROLES[rid]]
            })
        return total_power, details

    def check_constraints(self, t1_indices, include, expel, name_map):
        for grp in include:
            idxs = [name_map[f"{n}#{t}"] for n, t in grp]
            overlap = len(set(idxs).intersection(t1_indices))
            if overlap != 0 and overlap != len(idxs): return False
        for grp in expel:
            idxs = [name_map[f"{n}#{t}"] for n, t in grp]
            overlap = len(set(idxs).intersection(t1_indices))
            if overlap == 0 or overlap == len(idxs): return False
        return True

# ==========================================
# 4. 메인 실행 (Main Execution)
# ==========================================
def parse_manual_tier(tier_str):
    """
    입력 문자열(예: D3, G1, C)을 파싱하여 Tier, Division으로 반환
    """
    if not tier_str: return "Unranked", "0"
    
    tier_str = tier_str.upper()
    if tier_str in ["C", "GM", "M"]: # 챌린저, 그마, 마스터 (Division 1 고정)
        return tier_str, "1"
    
    if len(tier_str) >= 2:
        return tier_str[0], tier_str[1:] # D3 -> D, 3
    
    return "Unranked", "0"

def main():
    print("[ADC] System Initializing...")
    
    try:
        dm = DataManager()
        model = ADCModel()
    except Exception as e:
        print(f"[ERROR] Initialization Failed: {e}")
        return

    with open(FILE_INPUT, 'r', encoding='utf-8') as f:
        input_cfg = yaml.safe_load(f)
        
    players_input = input_cfg['players']
    inc_grp = input_cfg.get('include_group') or []
    exp_grp = input_cfg.get('expel_group') or []
    
    full_data = []
    name_map = {}
    
    print("[ADC] Loading Player Data...")
    
    for idx, p_info in enumerate(players_input):
        name, tag = p_info[0], p_info[1]
        
        # [기능 추가] 수동 티어 입력 확인
        manual_tier_str = None
        if len(p_info) >= 3:
            manual_tier_str = p_info[2]
            
        p_data = dm.find_in_db(name, tag)
        if not p_data:
            p_data = dm.fetch_player_data(name, tag)
        
        if not p_data:
            print(f"[WARN] Skipping invalid player: {name}#{tag}")
            continue
            
        # [기능 추가] 수동 티어가 있으면 데이터 오버라이드
        if manual_tier_str:
            m_tier, m_div = parse_manual_tier(manual_tier_str)
            p_data['tier'] = m_tier
            p_data['division'] = m_div
            # print(f"   -> Manual Tier Applied: {name} ({m_tier}{m_div})") # 디버깅용
            
        p_data['stats'] = model.calc_effective_power(p_data)
        full_data.append(p_data)
        name_map[f"{name}#{tag}"] = idx

    if len(full_data) < 10:
        print(f"[ERROR] Not enough players data (Count: {len(full_data)}/10).")
        return

    print("[ADC] Optimizing Team Balance...")
    
    indices = set(range(10))
    candidates = []
    all_combs = list(itertools.combinations(indices, 5))
    
    for i in range(len(all_combs) // 2):
        t1_idx = set(all_combs[i])
        t2_idx = indices - t1_idx
        
        if not model.check_constraints(t1_idx, inc_grp, exp_grp, name_map):
            continue
            
        team1 = [full_data[k] for k in t1_idx]
        team2 = [full_data[k] for k in t2_idx]
        
        pow1, det1 = model.solve_team_assignment(team1)
        pow2, det2 = model.solve_team_assignment(team2)
        
        diff = abs(pow1 - pow2)
        candidates.append((diff, {"pow": pow1, "det": det1}, {"pow": pow2, "det": det2}))

    if not candidates:
        print("[ERROR] No valid team combinations found.")
        return

    best = min(candidates, key=lambda x: x[0])
    
    print("\n" + "="*65)
    print(f"[RESULT] ADC Optimized Match (Diff: {best[0]:.4f})")
    print("="*65)
    
    def print_team_info(label, data):
        print(f"\n[{label}] Total Power: {data['pow']:.4f}")
        
        # 헤더 출력 (패딩 함수 적용)
        h_role = pad_text("Role", 5)
        h_name = pad_text("Name", 16)
        h_tier = pad_text("Tier", 6)
        h_pref = pad_text("Pref(L)", 9)
        h_power = "Power(S)"
        
        print(f"{h_role} | {h_name} | {h_tier} | {h_pref} | {h_power}")
        print("-" * 65)
        
        role_order = {r: i for i, r in enumerate(ROLES)}
        mems = sorted(data['det'], key=lambda x: role_order[x['role']])
        
        for m in mems:
            # 각 데이터 항목에 패딩 함수 적용
            v_role = pad_text(m['role'], 5)
            v_name = pad_text(m['name'], 16) # 닉네임 공간 16칸 확보
            v_tier = pad_text(m['tier'], 6)
            
            # 숫자는 그냥 f-string으로 처리 (길이 일정함)
            print(f"{v_role} | {v_name} | {v_tier} | {m['L']:.3f}     | {m['score']:.3f}")

    print_team_info("Blue Team", best[1])
    print_team_info("Red Team ", best[2])
    print("\n" + "="*65)

if __name__ == "__main__":
    main()