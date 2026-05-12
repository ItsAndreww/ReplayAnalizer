import json
import math
import pandas as pd

def keep_duplicates(ordered_pairs):
    d = {}
    for key, value in ordered_pairs:
        if key in d:
            if isinstance(d[key], list): d[key].append(value)
            else: d[key] = [d[key], value]
        else: d[key] = value
    return d

print("Завантажую JSON для створення AI датасету...")
with open('test.json', 'r', encoding='utf-16') as f:
    data = json.load(f, object_pairs_hook=keep_duplicates)

objects = data.get('objects', [])
frames = data.get('network_frames', {}).get('frames', [])

try:
    player_name_obj_id = objects.index("Engine.PlayerReplicationInfo:PlayerName")
    pri_obj_id = objects.index("Engine.Pawn:PlayerReplicationInfo")
    team_obj_id = objects.index("Engine.PlayerReplicationInfo:Team")
except ValueError:
    print("Помилка індексів об'єктів.")
    exit()

pri_to_name = {}
car_to_pri = {}
pri_to_team = {}

ball_actor_id = None
current_ball_loc = None
current_car_locs = {} 

raw_hits = [] 

print("Крок 1: Сканування всіх кадрів та збір станів поля...")
for frame_idx, frame in enumerate(frames):
    time = frame.get('time', 0.0)
    
    new_actors = frame.get('new_actors', [])
    if isinstance(new_actors, list):
        for actor in new_actors:
            object_id = actor.get('object_id')
            actor_id = actor.get('actor_id')
            if object_id is not None and object_id < len(objects):
                if 'Ball' in objects[object_id]:
                    ball_actor_id = actor_id

    updated_actors = frame.get('updated_actors', [])
    if not isinstance(updated_actors, list): continue
        
    for update in updated_actors:
        actor_id = update.get('actor_id')
        object_id = update.get('object_id')
        attribute = update.get('attribute', {})
        
        if object_id == player_name_obj_id:
            name = attribute.get('String')
            if name: pri_to_name[actor_id] = name
        elif object_id == pri_obj_id:
            pri_actor = attribute.get('ActiveActor', {}).get('actor') or attribute.get('FlaggedInt', {}).get('int')
            if pri_actor is not None: car_to_pri[actor_id] = pri_actor
        elif object_id == team_obj_id:
            team_actor = attribute.get('ActiveActor', {}).get('actor')
            if team_actor is not None: pri_to_team[actor_id] = team_actor
            
        if 'RigidBody' in attribute and attribute['RigidBody']:
            location = attribute['RigidBody'].get('location', {})
            if 'x' in location and 'y' in location:
                if actor_id == ball_actor_id:
                    current_ball_loc = location
                elif actor_id in car_to_pri:
                    current_car_locs[actor_id] = location

    if current_ball_loc is not None:
        for car_id, car_loc in current_car_locs.items():
            pri_id = car_to_pri.get(car_id)
            if not pri_id or pri_id not in pri_to_name: continue
                
            player_name = pri_to_name[pri_id]
            team_id = pri_to_team.get(pri_id)
            
            dx = car_loc['x'] - current_ball_loc['x']
            dy = car_loc['y'] - current_ball_loc['y']
            dz = car_loc.get('z', 0) - current_ball_loc.get('z', 0)
            distance_to_ball = math.sqrt(dx**2 + dy**2 + dz**2)
            
            if distance_to_ball < 250:
                snapshot = {
                    'Time': time,
                    'Player': player_name,
                    'Player_Car_ID': car_id,   # ФІКС 1: Зберігаємо ID нашої машини
                    'Team_ID': team_id,
                    'Ball_X': current_ball_loc['x'],
                    'Ball_Y': current_ball_loc['y'],
                    'Player_X': car_loc['x'],  # ФІКС 1: Координати гравця
                    'Player_Y': car_loc['y'],
                    'Car_Locs': current_car_locs.copy(), 
                    'Car_to_Pri': car_to_pri.copy(),
                    'Pri_to_Team': pri_to_team.copy()
                }
                raw_hits.append(snapshot)

print(f"Знайдено {len(raw_hits)} сирих дотиків. Фільтрую та витягую фічі...")

df_raw = pd.DataFrame(raw_hits)
ai_dataset = []

if not df_raw.empty:
    df_raw['Time_Diff'] = df_raw.groupby('Player')['Time'].diff()
    df_filtered = df_raw[(df_raw['Time_Diff'].isnull()) | (df_raw['Time_Diff'] > 0.5)].copy()
    df_filtered = df_filtered.sort_values(by='Time').reset_index(drop=True)

    # ФІКС 2: Динамічне визначення Синіх та Помаранчевих
    unique_teams = sorted(list(set(df_filtered['Team_ID'].dropna())))
    blue_team = unique_teams[0] if len(unique_teams) > 0 else None
    orange_team = unique_teams[1] if len(unique_teams) > 1 else None

    for idx, row in df_filtered.iterrows():
        hit_team = row['Team_ID']
        hit_car_id = row['Player_Car_ID']
        
        teammate_dists = []
        opponent_dists = []
        teammate_y_positions = [row['Player_Y']] 
        
        for other_car_id, other_loc in row['Car_Locs'].items():
            if other_car_id == hit_car_id: 
                continue # ФІКС 1: Пропускаємо самі себе!
            
            other_pri = row['Car_to_Pri'].get(other_car_id)
            if not other_pri: continue
            other_team = row['Pri_to_Team'].get(other_pri)
            
            # Рахуємо дистанцію від НАШОЇ МАШИНКИ до іншої
            dist = math.sqrt((other_loc['x'] - row['Player_X'])**2 + (other_loc['y'] - row['Player_Y'])**2)
            
            if other_team == hit_team:
                teammate_dists.append(dist)
                teammate_y_positions.append(other_loc['y'])
            else:
                opponent_dists.append(dist)
                
        nearest_tm = min(teammate_dists) if teammate_dists else 9999
        nearest_opp = min(opponent_dists) if opponent_dists else 9999
        
        # ФІКС 3: Коректний Last Man
        is_last_man = 0
        if len(teammate_y_positions) > 1:
            if hit_team == blue_team and row['Player_Y'] == min(teammate_y_positions):
                is_last_man = 1
            elif hit_team == orange_team and row['Player_Y'] == max(teammate_y_positions):
                is_last_man = 1

        possession_kept = 0 
        if idx + 1 < len(df_filtered):
            next_hit = df_filtered.iloc[idx + 1]
            if next_hit['Team_ID'] == hit_team:
                possession_kept = 1

        team_str = 'Blue' if hit_team == blue_team else 'Orange'

        ai_dataset.append({
            'Time': round(row['Time'], 2),
            'Player': row['Player'],
            'Team': team_str,
            'Ball_Y': round(row['Ball_Y'], 2),
            'Nearest_Teammate': round(nearest_tm, 2), 
            'Nearest_Opponent': round(nearest_opp, 2), 
            'Is_Last_Man': is_last_man, 
            'Possession_Kept': possession_kept 
        })

    final_df = pd.DataFrame(ai_dataset)
    final_df.to_csv('ai_features_dataset.csv', index=False)
    
    print(f"\nГотово! Зібрано {len(final_df)} ігрових ситуацій.")
    print("Файл 'ai_features_dataset.csv' збережено. Усі баги виправлено!")
else:
    print("Дотиків не знайдено.")