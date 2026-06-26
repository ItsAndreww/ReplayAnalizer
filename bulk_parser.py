import os
import json
import math
import pandas as pd
import numpy as np
import subprocess
import concurrent.futures

BOOST_PAD_POSITIONS = [
    (-3072, -4096), (3072, -4096),
    (-3584, 0),     (3584, 0),
    (-3072,  4096), (3072,  4096),
    (0, -4096),     (0, 4096),
    (-1792, -2944), (1792, -2944),
    (-1792,  2944), (1792,  2944),
    (-3584, -2560), (3584, -2560),
    (-3584,  2560), (3584,  2560),
]

# ТЕПЕР 13 УНІКАЛЬНИХ КЛАСІВ (Додано 12: Сейв)
ACTION_LABELS = {
    0: 'Втрата',
    1: 'Пас тімейту',
    2: 'Збір бусту',
    3: 'Удар по воротах',
    4: 'Клір',
    5: 'Бекборд пас',
    6: 'Дриблінг',
    7: 'Збереження позиції',
    8: 'Гольова ситуація',
    9: 'Фізичний тиск / Демо',
    10: 'Тіньовий захист',
    11: 'Кік-оф (Kickoff)',
    12: 'Сейв'
}

def keep_duplicates(ordered_pairs):
    d = {}
    for key, value in ordered_pairs:
        if key in d:
            if isinstance(d[key], list): d[key].append(value)
            else: d[key] = [d[key], value]
        else: d[key] = value
    return d

def nearest_boost_dist(px, py):
    return min(math.sqrt((px - bx)**2 + (py - by)**2) for bx, by in BOOST_PAD_POSITIONS)

def calculate_time_to_ball(px, py, pz, bx, by, bz, p_speed):
    dist = math.sqrt((px - bx)**2 + (py - by)**2 + (pz - bz)**2)
    return dist / (p_speed + 1)

def detect_mechanics(row, prev_row, next_row, hit_team, blue_team):
    bx, by, bz = row.get('Ball_X',0), row.get('Ball_Y',0), row.get('Ball_Z',0)
    px, py, pz = row.get('Player_X',0), row.get('Player_Y',0), row.get('Player_Z',0)
    pvx, pvy, pvz = row.get('Player_VX',0), row.get('Player_VY',0), row.get('Player_VZ',0)
    bvx, bvy, bvz = row.get('Ball_VX',0), row.get('Ball_VY',0), row.get('Ball_VZ',0)

    player_speed = math.sqrt(pvx**2 + pvy**2 + pvz**2)
    ball_speed   = math.sqrt(bvx**2 + bvy**2 + bvz**2)

    own_goal_y = -5120 if hit_team == blue_team else 5120
    dist_own = math.sqrt(px**2 + (py - own_goal_y)**2)

    is_on_ground    = pz < 120
    is_on_wall      = (abs(px) > 3800 or abs(py) > 4900) and pz > 100
    is_on_ceiling   = pz > 1900
    is_low_aerial   = 120 < pz <= 500
    is_mid_aerial   = 500 < pz <= 1200
    is_high_aerial  = pz > 1200

    ball_is_aerial  = bz > 300
    ball_on_wall    = (abs(bx) > 3800 or abs(by) > 4900) and bz > 100
    ball_on_ceiling = bz > 1900

    ball_directly_above = (abs(bx - px) < 200 and abs(by - py) < 200 and 50 < (bz - pz) < 300)

    prev_ball_speed = 0
    dot_dir = 0
    if prev_row is not None:
        pbvx, pbvy, pbvz = prev_row.get('Ball_VX',0), prev_row.get('Ball_VY',0), prev_row.get('Ball_VZ',0)
        prev_ball_speed = math.sqrt(pbvx**2 + pbvy**2 + pbvz**2)
        dot_dir = pvx * prev_row.get('Player_VX',0) + pvy * prev_row.get('Player_VY',0)

    is_flick = ball_directly_above and ball_speed > prev_ball_speed * 1.5 and ball_speed > 800
    is_aerial_hit = is_mid_aerial or is_high_aerial

    ball_was_going_to_own_goal = (
        (hit_team == blue_team and bvy < -300 and by < -2000) or
        (hit_team != blue_team and bvy >  300 and by >  2000)
    )
    is_save = ball_was_going_to_own_goal and ball_speed > 400
    is_wall_save = is_save and is_on_wall

    is_pinch = (ball_on_wall or ball_on_ceiling) and ball_speed > 1500 and player_speed > 800
    is_ground_pinch = bz < 150 and pz < 150 and pvz < -100 and ball_speed > 2000
    is_kuxir_pinch = is_pinch and abs(bx) > 3600 and abs(by) > 2000

    is_air_dribble = is_aerial_hit and ball_is_aerial and ball_speed < 1200 and player_speed > 600 and abs(bz - pz) < 400
    
    nearest_opp = row.get('Nearest_Opponent', 9999)
    is_fifty_fifty = nearest_opp < 400 and player_speed > 500 and ball_speed > 300
    is_supersonic = player_speed > 2200
    is_kickoff_touch = abs(bx) < 400 and abs(by) < 400 and player_speed > 1000 and abs(bz) < 200

    horizontal_speed = math.sqrt(pvx**2 + pvy**2)
    is_high_speed_ground = is_on_ground and horizontal_speed > 1800
    is_recovery = is_on_ground and horizontal_speed > 1000 and abs(pvx) > abs(pvy) * 2

    is_flip_reset_attempt = is_aerial_hit and abs(bz - pz) < 200 and player_speed > 800
    is_psycho = abs(by - own_goal_y) < 2000 and ball_speed > 1800 and ((hit_team == blue_team and bvy > 800) or (hit_team != blue_team and bvy < -800))
    is_fake = dot_dir < -40000 and player_speed > 400

    is_redirect = False
    if prev_row is not None and is_aerial_hit and prev_ball_speed > 1200 and ball_speed > 1200:
        dot_ball = (bvx * prev_row.get('Ball_VX',0) + bvy * prev_row.get('Ball_VY',0))
        if dot_ball < (prev_ball_speed * ball_speed * 0.85): 
            is_redirect = True

    is_shadowing = (
        abs(py - own_goal_y) < abs(by - own_goal_y) and
        ((hit_team == blue_team and pvy < -200 and bvy < -200) or
         (hit_team != blue_team and pvy > 200 and bvy > 200))
    )

    is_pre_jump = is_aerial_hit and nearest_opp > 1000 and ball_speed > 1000 and prev_ball_speed > 1000
    is_demo_attempt = is_supersonic and nearest_opp < 500
    is_boost_starve = row.get('Boost', 0) > 80 and nearest_boost_dist(px, py) < 300 and dist_own > 4000

    return {
        'Mech_On_Ground':    int(is_on_ground),
        'Mech_On_Wall':      int(is_on_wall),
        'Mech_On_Ceiling':   int(is_on_ceiling),
        'Mech_Low_Aerial':   int(is_low_aerial),
        'Mech_Mid_Aerial':   int(is_mid_aerial),
        'Mech_High_Aerial':  int(is_high_aerial),
        'Mech_Is_Flick':          int(is_flick),
        'Mech_Is_Save':           int(is_save),
        'Mech_Is_Wall_Save':      int(is_wall_save),
        'Mech_Is_Pinch':          int(is_pinch),
        'Mech_Is_Ground_Pinch':   int(is_ground_pinch),
        'Mech_Is_Kuxir_Pinch':    int(is_kuxir_pinch),
        'Mech_Is_Air_Dribble':    int(is_air_dribble),
        'Mech_Is_50_50':          int(is_fifty_fifty),
        'Mech_Is_Supersonic':     int(is_supersonic),
        'Mech_High_Speed_Ground': int(is_high_speed_ground),
        'Mech_Is_Recovery':       int(is_recovery),
        'Mech_Flip_Reset':        int(is_flip_reset_attempt),
        'Mech_Is_Psycho':         int(is_psycho),
        'Mech_Is_Fake':           int(is_fake),
        'Mech_Is_Redirect':       int(is_redirect),
        'Mech_Is_Shadowing':      int(is_shadowing),
        'Mech_Is_Pre_Jump':       int(is_pre_jump),
        'Mech_Is_Demo_Attempt':   int(is_demo_attempt),
        'Mech_Is_Boost_Starve':   int(is_boost_starve),
        'Mech_Is_Kickoff_Touch':  int(is_kickoff_touch),

        'Mech_Player_Z':          round(pz, 1),
        'Mech_Ball_Z_Relative':   round(bz - pz, 1),
        'Mech_Horiz_Speed':       round(horizontal_speed, 1),
        'Mech_Vertical_Vel':      round(pvz, 1),
    }

def classify_action(idx, df_filtered, row, hit_team, blue_team, orange_team, mech):
    bx, by, bz = row.get('Ball_X',0), row.get('Ball_Y',0), row.get('Ball_Z',0)
    px, py = row.get('Player_X',0), row.get('Player_Y',0)
    bvx, bvy, bvz = row.get('Ball_VX',0), row.get('Ball_VY',0), row.get('Ball_VZ',0)
    
    ball_speed = math.sqrt(bvx**2 + bvy**2 + bvz**2)
    own_goal_y = -5120 if hit_team == blue_team else  5120
    opp_goal_y =  5120 if hit_team == blue_team else -5120

    next_row = df_filtered.iloc[idx + 1] if idx + 1 < len(df_filtered) else None

    if mech.get('Mech_Is_Kickoff_Touch'): return 11
    
    # 12. СЕЙВ (Тепер окремо від Кліру)
    if mech.get('Mech_Is_Save') or mech.get('Mech_Is_Wall_Save'): return 12

    if mech.get('Mech_Is_Shadowing') and abs(by - opp_goal_y) > 3000: return 10
    if mech.get('Mech_Is_Demo_Attempt') or mech.get('Mech_Is_Boost_Starve'): return 9

    ball_in_opp_goal_zone = (hit_team == blue_team and by > 3600) or (hit_team == orange_team and by < -3600)
    player_in_attack = (hit_team == blue_team and py > 2500) or (hit_team == orange_team and py < -2500)
    ball_towards_opp_goal = (hit_team == blue_team and bvy > 0) or (hit_team == orange_team and bvy < 0)
    angle_to_goal = abs(bx) / max(abs(opp_goal_y - by), 1)
    
    no_tm_nearby = row.get('Nearest_Teammate', 9999) > 1000

    if ball_in_opp_goal_zone and player_in_attack and angle_to_goal < 0.75 and ball_towards_opp_goal and no_tm_nearby:
        return 8

    if nearest_boost_dist(px, py) < 1200 and row.get('Boost', 100) < 30:
        if next_row is not None and next_row['Player'] == row['Player'] and next_row.get('Boost', 0) > row.get('Boost', 0) + 20:
            return 2

    if next_row is not None and next_row['Player'] != row['Player'] and next_row.get('Team_ID') == hit_team and (next_row['Time'] - row['Time']) < 4.0:
        if ball_speed > 600:
            if (hit_team == blue_team and by > 3500 and bvz > 300) or (hit_team == orange_team and by < -3500 and bvz > 300):
                return 5
            return 1

    going_to_opp = (hit_team == blue_team and bvy > 500) or (hit_team == orange_team and bvy < -500)
    in_attack = (hit_team == blue_team and by > 2000) or (hit_team == orange_team and by < -2000)

    if going_to_opp and in_attack and ball_speed > 800 and angle_to_goal < 0.6:
        return 3

    # 4. КЛІР (Звичайний, без сейву)
    in_defense = (hit_team == blue_team and by < -2500) or (hit_team == orange_team and by > 2500)
    going_away_own = (hit_team == blue_team and bvy > 300) or (hit_team == orange_team and bvy < -300)

    if in_defense and going_away_own and ball_speed > 500:
        return 4

    if next_row is not None and next_row['Player'] == row['Player'] and (next_row['Time'] - row['Time']) < 1.5:
        if row.get('Nearest_Opponent', 9999) < 1500:
            return 6

    if next_row is not None and next_row.get('Team_ID') != hit_team:
        return 0

    return 7

def process_single_replay(replay_path):
    exe_path = os.path.join(os.getcwd(), "rrrocket.exe")
    result = subprocess.run([exe_path, "-n", replay_path], capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0: return []

    try: data = json.loads(result.stdout, object_pairs_hook=keep_duplicates)
    except: return []

    objects = data.get('objects', [])
    frames  = data.get('network_frames', {}).get('frames', [])

    def find_obj(name):
        for i, o in enumerate(objects):
            if name in o and "Default__" not in o: return i
        return -1

    player_name_obj_id = find_obj("PlayerReplicationInfo:PlayerName")
    pri_obj_id         = find_obj("Pawn:PlayerReplicationInfo")
    team_obj_id        = find_obj("PlayerReplicationInfo:Team")
    veh_obj_id         = find_obj("CarComponent_TA:Vehicle")
    boost_amt_obj_id   = find_obj("ReplicatedBoostAmount")

    if player_name_obj_id < 0 or pri_obj_id < 0: return []

    pri_to_name, car_to_pri, pri_to_team = {}, {}, {}
    boost_to_car, car_boost_levels = {}, {}
    ball_actor_id = None
    
    current_ball_loc, current_ball_vel = {}, {}
    current_car_locs, current_car_vels = {}, {}
    raw_hits = []

    # ТРЕКІНГ РАХУНКУ (Score_Diff Engine)
    cbs, cos, gcd = 0, 0, 0

    for frame_idx, frame in enumerate(frames):
        time = frame.get('time', 0.0)

        for actor in (frame.get('new_actors') or []):
            oid = actor.get('object_id')
            if oid is not None and oid < len(objects) and 'Ball' in objects[oid]:
                ball_actor_id = actor.get('actor_id')

        updated = frame.get('updated_actors') or []
        if not isinstance(updated, list): continue

        for upd in updated:
            aid, oid, attr = upd.get('actor_id'), upd.get('object_id'), upd.get('attribute', {})

            if oid == player_name_obj_id:
                name = attr.get('String')
                if name: pri_to_name[aid] = name

            if oid == pri_obj_id:
                pa = attr.get('ActiveActor', {}).get('actor') or attr.get('FlaggedInt', {}).get('int')
                if pa is not None: car_to_pri[aid] = pa

            if oid == team_obj_id:
                ta = attr.get('ActiveActor', {}).get('actor')
                if ta is not None: pri_to_team[aid] = ta

            if oid == veh_obj_id:
                ca = attr.get('ActiveActor', {}).get('actor')
                if ca: boost_to_car[aid] = ca

            if oid == boost_amt_obj_id:
                val = attr.get('Byte')
                if val is not None:
                    ca = boost_to_car.get(aid)
                    if ca: car_boost_levels[ca] = round((val / 255.0) * 100, 1)

            if 'ReplicatedBoost' in attr:
                b = attr['ReplicatedBoost']
                if isinstance(b, dict) and 'boost_amount' in b:
                    ca = boost_to_car.get(aid) or aid
                    car_boost_levels[ca] = round((b['boost_amount'] / 255.0) * 100, 1)

            if 'RigidBody' in attr and attr['RigidBody']:
                rb = attr['RigidBody']
                loc, vel = rb.get('location') or {}, rb.get('linear_velocity') or {}

                if aid == ball_actor_id:
                    current_ball_loc, current_ball_vel = loc, vel
                else:
                    current_car_locs[aid], current_car_vels[aid] = loc, vel

        if not current_ball_loc: continue

        # Логіка голів для Score_Diff
        by = current_ball_loc.get('y', 0)
        bx = current_ball_loc.get('x', 0)
        if abs(by) > 5140 and abs(bx) < 900:
            if gcd <= 0:
                if by > 0: cbs += 1
                else: cos += 1
                gcd = 90
        if gcd > 0: gcd -= 1

        for car_id, car_loc in current_car_locs.items():
            pri_id = car_to_pri.get(car_id)
            if not pri_id or pri_id not in pri_to_name: continue

            dx = car_loc.get('x', 0) - bx
            dy = car_loc.get('y', 0) - by
            dz = car_loc.get('z', 0) - current_ball_loc.get('z', 0)

            if math.sqrt(dx**2 + dy**2 + dz**2) < 250:
                raw_hits.append({
                    'Time': time, 'Player': pri_to_name[pri_id],
                    'Player_Car_ID': car_id, 'Team_ID': pri_to_team.get(pri_id),
                    'Ball_X': bx, 'Ball_Y': by, 'Ball_Z': current_ball_loc.get('z', 0),
                    'Ball_VX': current_ball_vel.get('x', 0), 'Ball_VY': current_ball_vel.get('y', 0), 'Ball_VZ': current_ball_vel.get('z', 0),
                    'Player_X': car_loc.get('x', 0), 'Player_Y': car_loc.get('y', 0), 'Player_Z': car_loc.get('z', 0),
                    'Player_VX': current_car_vels.get(car_id, {}).get('x', 0),
                    'Player_VY': current_car_vels.get(car_id, {}).get('y', 0),
                    'Player_VZ': current_car_vels.get(car_id, {}).get('z', 0),
                    'Boost': car_boost_levels.get(car_id, 33.0),
                    'Score_Diff_Blue': cbs - cos,
                    'Car_Locs': current_car_locs.copy(), 'Car_to_Pri': car_to_pri.copy(), 'Pri_to_Team': pri_to_team.copy(),
                })

    if not raw_hits: return []

    df_raw = pd.DataFrame(raw_hits)
    df_raw['Time_Diff'] = df_raw.groupby('Player')['Time'].diff()
    df_filtered = df_raw[df_raw['Time_Diff'].isnull() | (df_raw['Time_Diff'] > 0.5)].copy().sort_values('Time').reset_index(drop=True)

    unique_teams = sorted(df_filtered['Team_ID'].dropna().unique().tolist())
    blue_team   = unique_teams[0] if len(unique_teams) > 0 else None
    orange_team = unique_teams[1] if len(unique_teams) > 1 else None

    ai_dataset = []

    for idx, row in df_filtered.iterrows():
        hit_team = row.get('Team_ID')
        hit_car  = row.get('Player_Car_ID')

        bx, by, bz = row.get('Ball_X',0), row.get('Ball_Y',0), row.get('Ball_Z',0)
        px, py, pz = row.get('Player_X',0), row.get('Player_Y',0), row.get('Player_Z',0)
        pvx, pvy, pvz = row.get('Player_VX',0), row.get('Player_VY',0), row.get('Player_VZ',0)
        bvx, bvy, bvz = row.get('Ball_VX',0), row.get('Ball_VY',0), row.get('Ball_VZ',0)

        own_goal_y = -5120 if hit_team == blue_team else 5120
        opp_goal_y =  5120 if hit_team == blue_team else -5120

        dist_own = math.sqrt(px**2 + (py - own_goal_y)**2)
        dist_opp = math.sqrt(px**2 + (py - opp_goal_y)**2)
        dist_ball_own = math.sqrt(bx**2 + (by - own_goal_y)**2)
        dist_ball_opp = math.sqrt(bx**2 + (by - opp_goal_y)**2)
        angle_ball_own = abs(bx) / max(abs(own_goal_y - by), 1)

        player_speed = math.sqrt(pvx**2 + pvy**2 + pvz**2)
        ball_speed   = math.sqrt(bvx**2 + bvy**2 + bvz**2)

        nearest_boost = nearest_boost_dist(px, py)
        ttb_player = calculate_time_to_ball(px, py, pz, bx, by, bz, player_speed)

        tm_dists, op_dists, opp_ball_dists = [], [], []
        tm_y_positions = [py]
        teammates_behind, teammates_ahead, opp_blocking = 0, 0, 0
        passing_lane_open = 1

        for other_car, other_loc in row.get('Car_Locs', {}).items():
            if other_car == hit_car: continue
            other_pri = row['Car_to_Pri'].get(other_car)
            if not other_pri: continue
            other_team = row['Pri_to_Team'].get(other_pri)

            ox, oy, oz = other_loc.get('x', 0), other_loc.get('y', 0), other_loc.get('z', 0)
            d_player = math.sqrt((ox - px)**2 + (oy - py)**2)
            d_ball = math.sqrt((ox - bx)**2 + (oy - by)**2 + (oz - bz)**2)

            if other_team == hit_team:
                tm_dists.append(d_player)
                tm_y_positions.append(oy)
                if (hit_team == blue_team and oy < by) or (hit_team == orange_team and oy > by): teammates_behind += 1
                if (hit_team == blue_team and oy > by) or (hit_team == orange_team and oy < by): teammates_ahead += 1
            else:
                op_dists.append(d_player)
                opp_ball_dists.append(d_ball)
                if (hit_team == blue_team and oy < py and abs(ox - px) < 800) or (hit_team == orange_team and oy > py and abs(ox - px) < 800):
                    opp_blocking = 1
                if len(tm_dists) > 0 and d_player < min(tm_dists):
                    passing_lane_open = 0

        nearest_tm = tm_dists[0] if tm_dists else 9999
        nearest_op = min(op_dists) if op_dists else 9999
        opp_to_ball = min(opp_ball_dists) if opp_ball_dists else 9999

        is_last_man = 0
        if len(tm_y_positions) > 1:
            if hit_team == blue_team and py == min(tm_y_positions): is_last_man = 1
            if hit_team == orange_team and py == max(tm_y_positions): is_last_man = 1

        row['Nearest_Opponent'] = nearest_op
        row['Nearest_Teammate'] = nearest_tm

        prev_row_data = df_filtered.iloc[idx - 1] if idx > 0 else None
        next_row_data = df_filtered.iloc[idx + 1] if idx + 1 < len(df_filtered) else None
        
        mechanics = detect_mechanics(row, prev_row_data, next_row_data, hit_team, blue_team)
        action_label = classify_action(idx, df_filtered, row, hit_team, blue_team, orange_team, mechanics)

        # ТАКТИЧНІ ФІЧІ ЧАСУ ТА РАХУНКУ
        time_remaining = max(0, 300 - row['Time'])
        is_overtime = 1 if row['Time'] > 300 else 0
        score_diff = row.get('Score_Diff_Blue', 0) if hit_team == blue_team else -row.get('Score_Diff_Blue', 0)

        ai_dataset.append({
            'Time': round(row['Time'], 2), 'Player': row['Player'], 'Team': 'Blue' if hit_team == blue_team else 'Orange',
            'Ball_X': round(bx, 1), 'Ball_Y': round(by, 1), 'Ball_Z': round(bz, 1),
            'Player_X': round(px, 1), 'Player_Y': round(py, 1), 'Player_Z': round(pz, 1),
            
            'Player_Speed': round(player_speed, 1), 'Ball_Speed': round(ball_speed, 1),
            'Player_VX': round(pvx, 1), 'Player_VY': round(pvy, 1), 'Ball_VX': round(bvx, 1), 'Ball_VY': round(bvy, 1), 'Ball_VZ': round(bvz, 1),

            'Dist_to_Own_Goal': round(dist_own, 1), 'Dist_to_Opp_Goal': round(dist_opp, 1),
            'Dist_Ball_to_Own_Goal': round(dist_ball_own, 1), 'Dist_Ball_to_Opp_Goal': round(dist_ball_opp, 1),
            'Angle_Ball_to_Own_Goal': round(angle_ball_own, 4),

            'Nearest_Teammate': round(nearest_tm, 1), 'Nearest_Opponent': round(nearest_op, 1), 'Opp_Dist_To_Ball': round(opp_to_ball, 1),
            'Teammates_Ahead': teammates_ahead, 'Teammates_Behind_Ball': teammates_behind,
            'Is_Last_Man': is_last_man, 'Opp_Blocking': opp_blocking,

            'Boost_Amount': round(row.get('Boost', 33), 1), 'Nearest_Boost_Pad': round(nearest_boost, 1),
            
            'Tactics_Time_To_Ball': round(ttb_player, 2),
            'Tactics_Passing_Lane_Open': passing_lane_open,
            'Time_Remaining': round(time_remaining, 1),
            'Is_Overtime': is_overtime,
            'Score_Diff': score_diff,

            **mechanics,
            'Action_Label': action_label,
            'Action_Name': ACTION_LABELS[action_label],
        })

    return ai_dataset


if __name__ == "__main__":
    replay_folder = "rlcs_replays"
    all_data = []

    if not os.path.exists(replay_folder):
        print(f"❌ Папка '{replay_folder}' не знайдена!")
    else:
        files = [os.path.join(replay_folder, f) for f in os.listdir(replay_folder) if f.endswith(".replay")]
        total_files = len(files)
        print(f"🚀 Починаємо обробку {total_files} реплеїв на всіх ядрах...\n")

        processed_count = 0

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_path = {executor.submit(process_single_replay, path): path for path in files}
            
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                processed_count += 1
                
                try:
                    rows = future.result()
                    if rows:
                        all_data.extend(rows)
                        print(f"[{processed_count}/{total_files}] ✅ {os.path.basename(path)}: +{len(rows)} ситуацій")
                    else:
                        print(f"[{processed_count}/{total_files}] ⚠️ {os.path.basename(path)}: пропущено (немає даних)")
                except Exception as exc:
                    print(f"[{processed_count}/{total_files}] ❌ {os.path.basename(path)}: виняток: {exc}")

        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv('ai_features_dataset_v4.csv', index=False)
            print(f"\n🎉 ГОТОВО! Проаналізовано {processed_count} матчів.")
            print(f"✅ Датасет збережено: {len(df)} ситуацій загалом.")