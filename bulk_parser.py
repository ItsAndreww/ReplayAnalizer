import os
import json
import math
import pandas as pd
import subprocess

def keep_duplicates(ordered_pairs):
    d = {}
    for key, value in ordered_pairs:
        if key in d:
            if isinstance(d[key], list): d[key].append(value)
            else: d[key] = [d[key], value]
        else: d[key] = value
    return d

def process_single_replay(replay_path):
    exe_path = os.path.join(os.getcwd(), "rrrocket.exe")
    
    result = subprocess.run([exe_path, "-n", replay_path], capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0: return []

    try:
        data = json.loads(result.stdout, object_pairs_hook=keep_duplicates)
    except:
        return []

    objects = data.get('objects', [])
    frames = data.get('network_frames', {}).get('frames', [])

    try:
        player_name_obj_id = objects.index("Engine.PlayerReplicationInfo:PlayerName")
        pri_obj_id = objects.index("Engine.Pawn:PlayerReplicationInfo")
        team_obj_id = objects.index("Engine.PlayerReplicationInfo:Team")
    except ValueError:
        return []

    # НОВЕ: Динамічний пошук ID для бусту
    veh_obj_id = -1
    boost_amt_obj_id = -1
    for i, obj in enumerate(objects):
        if 'CarComponent_TA:Vehicle' in obj: veh_obj_id = i
        if 'ReplicatedBoostAmount' in obj: boost_amt_obj_id = i

    pri_to_name, car_to_pri, pri_to_team = {}, {}, {}
    ball_actor_id, current_ball_loc = None, None
    current_car_locs = {}
    
    # НОВЕ: Словники для супер-фічей
    current_car_velocities = {}
    boost_to_car = {}
    car_boost_levels = {} 
    
    raw_hits = []

    for frame_idx, frame in enumerate(frames):
        time = frame.get('time', 0.0)
        
        new_actors = frame.get('new_actors', [])
        if isinstance(new_actors, list):
            for actor in new_actors:
                obj_id = actor.get('object_id')
                act_id = actor.get('actor_id')
                if obj_id is not None and obj_id < len(objects) and 'Ball' in objects[obj_id]:
                    ball_actor_id = act_id

        updated_actors = frame.get('updated_actors', [])
        if not isinstance(updated_actors, list): continue
            
        for update in updated_actors:
            act_id = update.get('actor_id')
            obj_id = update.get('object_id')
            attribute = update.get('attribute', {})
            
            if obj_id == player_name_obj_id:
                name = attribute.get('String')
                if name: pri_to_name[act_id] = name
            elif obj_id == pri_obj_id:
                pri_actor = attribute.get('ActiveActor', {}).get('actor') or attribute.get('FlaggedInt', {}).get('int')
                if pri_actor is not None: car_to_pri[act_id] = pri_actor
            elif obj_id == team_obj_id:
                team_actor = attribute.get('ActiveActor', {}).get('actor')
                if team_actor is not None: pri_to_team[act_id] = team_actor
                
            # ЛОВИМО БУСТ!
            elif obj_id == veh_obj_id:
                car_actor = attribute.get('ActiveActor', {}).get('actor')
                if car_actor: boost_to_car[act_id] = car_actor
            elif obj_id == boost_amt_obj_id:
                val = attribute.get('Byte')
                if val is not None:
                    car_actor = boost_to_car.get(act_id)
                    if car_actor:
                        # Переводимо байти (0-255) у відсотки бусту (0-100)
                        car_boost_levels[car_actor] = round((val / 255.0) * 100, 1)

            # ЛОВИМО КООРДИНАТИ ТА ШВИДКІСТЬ!
            if 'RigidBody' in attribute and attribute['RigidBody']:
                # Додаємо "or {}", щоб якщо гра поверне None, ми отримали пустий словник
                loc = attribute['RigidBody'].get('location') or {}
                vel = attribute['RigidBody'].get('linear_velocity') or {}
                
                if 'x' in loc and 'y' in loc:
                    if act_id == ball_actor_id: 
                        current_ball_loc = loc
                    elif act_id in car_to_pri: 
                        current_car_locs[act_id] = loc
                        if 'x' in vel and 'y' in vel:
                            current_car_velocities[act_id] = vel

        if current_ball_loc is not None:
            for car_id, car_loc in current_car_locs.items():
                pri_id = car_to_pri.get(car_id)
                if not pri_id or pri_id not in pri_to_name: continue
                    
                dx = car_loc['x'] - current_ball_loc['x']
                dy = car_loc['y'] - current_ball_loc['y']
                dz = car_loc.get('z', 0) - current_ball_loc.get('z', 0)
                
                if math.sqrt(dx**2 + dy**2 + dz**2) < 250:
                    car_vel = current_car_velocities.get(car_id, {'x': 0, 'y': 0})
                    
                    raw_hits.append({
                        'Time': time, 'Player': pri_to_name[pri_id], 
                        'Player_Car_ID': car_id, 'Team_ID': pri_to_team.get(pri_id),
                        'Ball_X': current_ball_loc['x'], 'Ball_Y': current_ball_loc['y'],
                        'Player_X': car_loc['x'], 'Player_Y': car_loc['y'],
                        'Player_VX': car_vel.get('x', 0), 'Player_VY': car_vel.get('y', 0),
                        'Boost': car_boost_levels.get(car_id, 33.0), # Дефолтно 33, якщо буст ще не оновився
                        'Car_Locs': current_car_locs.copy(), 'Car_to_Pri': car_to_pri.copy(), 'Pri_to_Team': pri_to_team.copy()
                    })

    df_raw = pd.DataFrame(raw_hits)
    ai_dataset = []

    if not df_raw.empty:
        df_raw['Time_Diff'] = df_raw.groupby('Player')['Time'].diff()
        df_filtered = df_raw[(df_raw['Time_Diff'].isnull()) | (df_raw['Time_Diff'] > 0.5)].copy()
        df_filtered = df_filtered.sort_values(by='Time').reset_index(drop=True)

        unique_teams = sorted(list(set(df_filtered['Team_ID'].dropna())))
        blue_team = unique_teams[0] if len(unique_teams) > 0 else None
        orange_team = unique_teams[1] if len(unique_teams) > 1 else None

        for idx, row in df_filtered.iterrows():
            hit_team = row['Team_ID']
            hit_car_id = row['Player_Car_ID']
            teammate_dists, opponent_dists = [], []
            teammate_y_positions = [row['Player_Y']] 
            
            for other_car_id, other_loc in row['Car_Locs'].items():
                if other_car_id == hit_car_id: continue
                other_pri = row['Car_to_Pri'].get(other_car_id)
                if not other_pri: continue
                other_team = row['Pri_to_Team'].get(other_pri)
                
                dist = math.sqrt((other_loc['x'] - row['Player_X'])**2 + (other_loc['y'] - row['Player_Y'])**2)
                
                if other_team == hit_team:
                    teammate_dists.append(dist)
                    teammate_y_positions.append(other_loc['y'])
                else: opponent_dists.append(dist)
                    
            nearest_tm = min(teammate_dists) if teammate_dists else 9999
            nearest_opp = min(opponent_dists) if opponent_dists else 9999
            
            is_last_man = 0
            if len(teammate_y_positions) > 1:
                if hit_team == blue_team and row['Player_Y'] == min(teammate_y_positions): is_last_man = 1
                elif hit_team == orange_team and row['Player_Y'] == max(teammate_y_positions): is_last_man = 1

            possession_kept = 0 
            if idx + 1 < len(df_filtered):
                if df_filtered.iloc[idx + 1]['Team_ID'] == hit_team: possession_kept = 1

            # НОВЕ: Рахуємо загальну швидкість машини
            player_speed = math.sqrt(row['Player_VX']**2 + row['Player_VY']**2)

            ai_dataset.append({
                'Time': round(row['Time'], 2),
                'Player': row['Player'],
                'Team': 'Blue' if hit_team == blue_team else 'Orange',
                'Ball_Y': round(row['Ball_Y'], 2),
                'Nearest_Teammate': round(nearest_tm, 2), 
                'Nearest_Opponent': round(nearest_opp, 2), 
                'Is_Last_Man': is_last_man, 
                'Boost_Amount': round(row['Boost'], 1),      # ФІЧА: ЖАДІБНІСТЬ
                'Player_Speed': round(player_speed, 1),      # ФІЧА: ІМПУЛЬС
                'Player_VY': round(row['Player_VY'], 1),     # ФІЧА: РОТАЦІЯ (куди їхав)
                'Player_Y': round(row['Player_Y'], 1),       # ФІЧА: ПОЗИЦІЯ
                'Possession_Kept': possession_kept 
            })

    return ai_dataset

# ==========================================
# ГОЛОВНИЙ ЦИКЛ ОБРОБКИ ПАПКИ
# ==========================================
replay_folder = "rlcs_replays"
all_matches_data = []

if not os.path.exists(replay_folder):
    print(f"Папка {replay_folder} не знайдена!")
else:
    replay_files = [f for f in os.listdir(replay_folder) if f.endswith(".replay")]
    print(f"🚀 Починаю масову обробку {len(replay_files)} реплеїв з АНАЛІЗОМ БУСТУ ТА РОТАЦІЇ...\n")
    
    for i, filename in enumerate(replay_files):
        filepath = os.path.join(replay_folder, filename)
        print(f"[{i+1}/{len(replay_files)}] Аналізую: {filename}...")
        
        match_data = process_single_replay(filepath)
        if match_data:
            all_matches_data.extend(match_data)
        else:
            print("   -> Помилка парсингу або пустий реплей.")

    if all_matches_data:
        final_df = pd.DataFrame(all_matches_data)
        final_df.to_csv('ai_features_dataset_v2.csv', index=False) # ЗБЕРІГАЄМО ЯК V2
        print(f"\n🎉 ГОТОВО! Датасет V2 на {len(final_df)} ситуацій збережено у 'ai_features_dataset_v2.csv'!")
        print("Наступний крок: Оновити train_ai.py, щоб він навчився розуміти Буст і Швидкість!")
    else:
        print("\n❌ Не вдалося зібрати дані з реплеїв.")