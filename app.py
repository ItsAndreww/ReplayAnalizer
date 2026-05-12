import json
import math
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import subprocess
import os

# --- 1. ФУНКЦІЇ ПАРСИНГУ ТА ОБРОБКИ ---

def parse_uploaded_replay(uploaded_file):
    temp_replay_path = "temp_uploaded.replay"
    with open(temp_replay_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    exe_path = os.path.join(os.getcwd(), "rrrocket.exe") 
    
    result = subprocess.run(
        [exe_path, "-n", temp_replay_path],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    if os.path.exists(temp_replay_path):
        os.remove(temp_replay_path)

    if result.returncode == 0:
        return json.loads(result.stdout, object_pairs_hook=keep_duplicates)
    else:
        st.error(f"Помилка rrrocket: {result.stderr}")
        return None

st.set_page_config(page_title="RL Analytics Dashboard", layout="wide")

def keep_duplicates(ordered_pairs):
    d = {}
    for key, value in ordered_pairs:
        if key in d:
            if isinstance(d[key], list): d[key].append(value)
            else: d[key] = [d[key], value]
        else: d[key] = value
    return d

def process_match_data(data):
    objects = data.get('objects', [])
    net_frames = data.get('network_frames') or {}
    frames = net_frames.get('frames', [])
    properties = data.get('properties', {})
    
    match_stats = []
    known_names = []
    raw_stats = properties.get('PlayerStats', [])
    if not isinstance(raw_stats, list): raw_stats = [raw_stats]
    
    for player in raw_stats:
        if isinstance(player, dict):
            p_name = player.get('Name', 'Unknown')
            known_names.append(p_name)
            match_stats.append({
                'Name': p_name, 'Score': player.get('Score', 0),
                'Goals': player.get('Goals', 0), 'Assists': player.get('Assists', 0),
                'Saves': player.get('Saves', 0), 'Shots': player.get('Shots', 0),
                'Demos': player.get('Demolishes', 0)
            })

    pri_to_name, car_to_pri, pri_to_team = {}, {}, {}
    comp_to_car = {}
    player_boost_data = {name: [] for name in known_names}
    player_paths = {name: {'x': [], 'y': []} for name in known_names}
    
    ball_y_timeline = []
    ball_actor_id = None
    
    # Оновлений словник для AI V2
    synced_data = {name: {'x': {}, 'y': {}, 'speed': {}, 'vy': {}, 'boost': {}} for name in known_names}
    ball_synced = {'x': {}, 'y': {}, 'vx': {}, 'vy': {}}

    # --- НАДІЙНИЙ ПОШУК (Без зависань) ---
    def get_exact_obj_id(obj_name):
        for i, o in enumerate(objects):
            if obj_name in o and "Default__" not in o:
                return i
        return None

    pri_obj_id = get_exact_obj_id("Pawn:PlayerReplicationInfo")
    team_obj_id = get_exact_obj_id("PlayerReplicationInfo:Team")
    vehicle_obj_id = get_exact_obj_id("CarComponent_TA:Vehicle")
    # -------------------------------------

    for frame_idx, frame in enumerate(frames):
        new_actors = frame.get('new_actors', [])
        if isinstance(new_actors, list):
            for actor in new_actors:
                oid = actor.get('object_id')
                if oid is not None and oid < len(objects) and 'Ball' in objects[oid]:
                    ball_actor_id = actor.get('actor_id')

        updated_actors = frame.get('updated_actors', [])
        if isinstance(updated_actors, list):
            for update in updated_actors:
                actor_id = update.get('actor_id')
                object_id = update.get('object_id')
                attribute = update.get('attribute', {})

                if "Reservation" in attribute: raw_name = attribute["Reservation"].get("name")
                elif "String" in attribute: raw_name = attribute["String"]
                else: raw_name = None

                if raw_name:
                    clean_raw = str(raw_name).strip().lower()
                    for real_name in known_names:
                        if str(real_name).strip().lower() in clean_raw:
                            pri_to_name[actor_id] = real_name
                            break

                if object_id == pri_obj_id:
                    pa = attribute.get('ActiveActor', {}).get('actor') or attribute.get('FlaggedInt', {}).get('int')
                    if pa is not None: car_to_pri[actor_id] = pa
                elif object_id == team_obj_id:
                    ta = attribute.get('ActiveActor', {}).get('actor')
                    if ta is not None: pri_to_team[actor_id] = ta
                elif object_id == vehicle_obj_id:
                    car_actor = attribute.get('ActiveActor', {}).get('actor') or attribute.get('FlaggedInt', {}).get('int')
                    if car_actor is not None: comp_to_car[actor_id] = car_actor 

                boost_pct = None
                if "ReplicatedBoost" in attribute:
                    b_info = attribute["ReplicatedBoost"]
                    if isinstance(b_info, dict) and "boost_amount" in b_info:
                        boost_pct = round((b_info["boost_amount"] / 255) * 100)
                elif "Byte" in attribute and object_id < len(objects) and "BoostAmount" in objects[object_id]:
                    boost_pct = round((attribute["Byte"] / 255) * 100)

                if boost_pct is not None:
                    p_name = None
                    if actor_id in comp_to_car:
                        car_id = comp_to_car[actor_id]
                        if car_id in car_to_pri and car_to_pri[car_id] in pri_to_name: p_name = pri_to_name[car_to_pri[car_id]]
                    elif actor_id in car_to_pri and car_to_pri[actor_id] in pri_to_name: p_name = pri_to_name[car_to_pri[actor_id]]
                    elif actor_id in pri_to_name: p_name = pri_to_name[actor_id]
                    
                    if p_name and p_name in player_boost_data: 
                        player_boost_data[p_name].append(boost_pct)
                        synced_data[p_name]['boost'][frame_idx] = boost_pct # <-- Зберігаємо буст для ШІ

                if 'RigidBody' in attribute and attribute['RigidBody']:
                    rb = attribute['RigidBody']
                    loc, vel = rb.get('location') or {}, rb.get('linear_velocity') or {}
                    
                    if actor_id == ball_actor_id:
                        if 'y' in loc: 
                            ball_y_timeline.append(loc['y'])
                            ball_synced['y'][frame_idx] = loc['y']
                        if 'x' in loc: ball_synced['x'][frame_idx] = loc['x']
                        if 'x' in vel and 'y' in vel:
                            ball_synced['vx'][frame_idx] = vel['x']
                            ball_synced['vy'][frame_idx] = vel['y']
                        
                    pid = car_to_pri.get(actor_id)
                    if pid and pid in pri_to_name:
                        pname = pri_to_name[pid]
                        if pname in known_names:
                            if 'x' in loc and 'y' in loc:
                                player_paths[pname]['x'].append(loc['x'])
                                player_paths[pname]['y'].append(loc['y'])
                                synced_data[pname]['x'][frame_idx] = loc['x']
                                synced_data[pname]['y'][frame_idx] = loc['y']
                                
                            if 'x' in vel and 'y' in vel and 'z' in vel:
                                sp = ((vel['x']**2 + vel['y']**2 + vel['z']**2)**0.5) * 0.036
                                synced_data[pname]['speed'][frame_idx] = sp
                                synced_data[pname]['vy'][frame_idx] = vel['y'] # <-- Напрямок для ШІ

    full_df = pd.DataFrame(index=range(len(frames)))
    if len(frames) > 0:
        for name in known_names:
            full_df[f"{name}_x"] = pd.to_numeric(pd.Series(synced_data[name]['x']).reindex(full_df.index).ffill(), errors='coerce')
            full_df[f"{name}_y"] = pd.to_numeric(pd.Series(synced_data[name]['y']).reindex(full_df.index).ffill(), errors='coerce')
            full_df[f"{name}_speed"] = pd.to_numeric(pd.Series(synced_data[name]['speed']).reindex(full_df.index).ffill(limit=120).fillna(0), errors='coerce')
            full_df[f"{name}_vy"] = pd.to_numeric(pd.Series(synced_data[name]['vy']).reindex(full_df.index).ffill().fillna(0), errors='coerce')
            full_df[f"{name}_boost"] = pd.to_numeric(pd.Series(synced_data[name]['boost']).reindex(full_df.index).ffill().fillna(33), errors='coerce')
            
        full_df["ball_x"] = pd.to_numeric(pd.Series(ball_synced['x']).reindex(full_df.index).ffill(), errors='coerce')
        full_df["ball_y"] = pd.to_numeric(pd.Series(ball_synced['y']).reindex(full_df.index).ffill(), errors='coerce')
        full_df["ball_vx"] = pd.to_numeric(pd.Series(ball_synced['vx']).reindex(full_df.index).ffill(), errors='coerce')
        full_df["ball_vy"] = pd.to_numeric(pd.Series(ball_synced['vy']).reindex(full_df.index).ffill(), errors='coerce')

    hits_data = []
    if not full_df.empty:
        for name in known_names:
            dist_to_ball = np.sqrt((full_df[f"{name}_x"] - full_df["ball_x"])**2 + (full_df[f"{name}_y"] - full_df["ball_y"])**2)
            hit_frames = full_df[dist_to_ball < 300].index.tolist()
            last_hit_frame = -999
            for frame in hit_frames:
                if frame - last_hit_frame > 30:
                    hits_data.append({'Player': name, 'Ball_X': full_df.loc[frame, "ball_x"], 'Ball_Y': full_df.loc[frame, "ball_y"], 'Frame': frame})
                    last_hit_frame = frame
    hits_df = pd.DataFrame(hits_data)

    unique_teams = sorted(list(set(pri_to_team.values())))
    p_colors = {name: 'royalblue' for name in known_names}
    for pri_id, p_name in pri_to_name.items():
        t_id = pri_to_team.get(pri_id)
        if p_name in p_colors and len(unique_teams) >= 2 and t_id == unique_teams[0]:
            p_colors[p_name] = 'darkorange'

    return player_paths, p_colors, match_stats, ball_y_timeline, player_boost_data, full_df, hits_df

def calculate_xg(x, y, team_color):
    target_y = 5120 if team_color == 'royalblue' else -5120
    if (team_color == 'royalblue' and y < 0) or (team_color == 'darkorange' and y > 0):
        return 0.0
    distance = math.sqrt(x**2 + (target_y - y)**2)
    dist_factor = max(0.0, 1.0 - (distance / 4500)**1.5)
    angle_factor = max(0.0, 1.0 - (abs(x) / 3500)**2)
    base_xg = dist_factor * angle_factor
    return max(0.01, min(0.95, base_xg))

# --- 2. ГОЛОВНИЙ ІНТЕРФЕЙС ---

st.title("RL Analytics Dashboard: Team UKI")

st.sidebar.header("📁 Завантаження")
uploaded_file = st.sidebar.file_uploader("Оберіть Replay файл", type=['replay'])

if uploaded_file:
    with st.spinner('Автоматичний парсинг реплею та генерація аналітики...'):
        json_data = parse_uploaded_replay(uploaded_file)
        
        if json_data:
            player_paths, player_colors, match_stats, ball_y, player_boost_data, full_df, hits_df = process_match_data(json_data)
            
            # --- РОЗРАХУНОК xG ТА AI COACH (V2) ---
            import joblib
            try:
                ai_coach = joblib.load('ai_coach_model_v2.pkl')
            except Exception as e:
                ai_coach = None
                st.warning("⚠️ Не вдалося завантажити ai_coach_model_v2.pkl. AI-Тренер буде вимкнений.")

            player_all_touches_xg = {}
            if not hits_df.empty:
                # Додаємо дефолтні колонки, щоб Pandas ніколи не крашився
                for col in ['Nearest_Teammate', 'Nearest_Opponent', 'Is_Last_Man', 'Boost_Amount', 'Player_Speed', 'Player_VY', 'Player_Y']:
                    hits_df[col] = 0.0

                for idx, row in hits_df.iterrows():
                    p_name = row['Player']
                    p_color = player_colors.get(p_name, 'cyan')
                    hit_xg = calculate_xg(row['Ball_X'], row['Ball_Y'], p_color)
                    hits_df.at[idx, 'xG'] = hit_xg
                    
                    if p_name not in player_all_touches_xg:
                        player_all_touches_xg[p_name] = []
                    player_all_touches_xg[p_name].append(hit_xg)

                    # --- ЗБІР ФІЧЕЙ ДЛЯ ML (V2) ---
                    frame = row['Frame']
                    hits_df.at[idx, 'Ball_Y'] = row['Ball_Y']

                    if f"{p_name}_x" in full_df.columns:
                        px = full_df.loc[frame, f"{p_name}_x"]
                        py = full_df.loc[frame, f"{p_name}_y"]
                        
                        hits_df.at[idx, 'Player_Y'] = py
                        hits_df.at[idx, 'Player_Speed'] = full_df.loc[frame, f"{p_name}_speed"]
                        hits_df.at[idx, 'Player_VY'] = full_df.loc[frame, f"{p_name}_vy"]
                        hits_df.at[idx, 'Boost_Amount'] = full_df.loc[frame, f"{p_name}_boost"]

                        teammate_dists, opponent_dists, teammate_y_pos = [], [], [py]

                        for other_p, other_c in player_colors.items():
                            if other_p == p_name or f"{other_p}_x" not in full_df.columns: continue
                            ox = full_df.loc[frame, f"{other_p}_x"]
                            oy = full_df.loc[frame, f"{other_p}_y"]
                            if pd.isna(ox) or pd.isna(oy): continue
                                
                            dist = math.sqrt((ox - px)**2 + (oy - py)**2)
                            if other_c == p_color:
                                teammate_dists.append(dist)
                                teammate_y_pos.append(oy)
                            else:
                                opponent_dists.append(dist)
                                
                        hits_df.at[idx, 'Nearest_Teammate'] = min(teammate_dists) if teammate_dists else 9999
                        hits_df.at[idx, 'Nearest_Opponent'] = min(opponent_dists) if opponent_dists else 9999
                        
                        is_last = 0
                        if len(teammate_y_pos) > 1:
                            if p_color == 'royalblue' and py == min(teammate_y_pos): is_last = 1
                            elif p_color == 'darkorange' and py == max(teammate_y_pos): is_last = 1
                        hits_df.at[idx, 'Is_Last_Man'] = is_last

            # --- ОЦІНКА ШІ (З ЗАПОБІЖНИКОМ) ---
            if ai_coach is not None and not hits_df.empty:
                try:
                    features = ['Ball_Y', 'Nearest_Teammate', 'Nearest_Opponent', 'Is_Last_Man', 'Boost_Amount', 'Player_Speed', 'Player_VY', 'Player_Y']
                    ai_features = hits_df[features].fillna(0)
                    hits_df['AI_Score'] = ai_coach.predict_proba(ai_features)[:, 1] * 100
                except Exception as e:
                    st.error(f"Помилка під час оцінки ШІ: {e}")

            # ==========================================
            # --- ОСЬ ЦЕЙ БЛОК МИ ВИПАДКОВО ЗАГУБИЛИ ---
            # ==========================================
            player_total_xg = {}
            for p in match_stats:
                name = p['Name']
                allowed_shots = max(p.get('Shots', 0), p.get('Goals', 0))
                
                if name in player_all_touches_xg and allowed_shots > 0:
                    top_touches = sorted(player_all_touches_xg[name], reverse=True)
                    player_total_xg[name] = sum(top_touches[:allowed_shots])
                else:
                    player_total_xg[name] = 0.0
            # ==========================================

            plt.style.use('dark_background')

            # ==========================================
            # 1. ТАБЛИЦЯ СТАТИСТИКИ
            # ==========================================
            st.subheader("📊 Рейтинг гравців (Rating 2.0) та xG")
            stats_data = []
            
            AVG_GOALS, AVG_ASSISTS, AVG_SAVES, AVG_SHOTS, AVG_DEMOS = 0.75, 0.60, 1.50, 3.00, 1.20
            
            for p in match_stats:
                name = p['Name']
                color = player_colors.get(name, 'cyan')
                team = "Сині" if color == 'royalblue' else "Помаранчеві"
                xg_val = player_total_xg.get(name, 0.0)
                goals, assists, saves, shots, demos = p['Goals'], p['Assists'], p['Saves'], p['Shots'], p['Demos']
                
                rating = 6
                if goals > 0:
                    rating += (goals * 0.8) + ((goals - xg_val) * 0.5)
                rating += (assists - AVG_ASSISTS) * 0.5 + (saves - AVG_SAVES) * 0.3 + (shots - AVG_SHOTS) * 0.1 + (demos - AVG_DEMOS) * 0.15
                if shots > 3 and goals == 0: rating -= 0.5
                final_rating = max(1.0, min(10.0, rating))

                stats_data.append({"Команда": team, "Гравець": name, "Rating 2.0": final_rating, "xG": xg_val, "Goals": goals, "Assists": assists, "Saves": saves, "Shots": shots, "Score": p['Score']})
            
            df = pd.DataFrame(stats_data)
            # Списки імен для заголовків
            blue_names = ", ".join(df[df["Команда"] == "Сині"]["Гравець"].tolist())
            orange_names = ", ".join(df[df["Команда"] == "Помаранчеві"]["Гравець"].tolist())

            blue_team = df[df["Команда"] == "Сині"].drop(columns=["Команда"]).sort_values(by="Rating 2.0", ascending=False)
            orange_team = df[df["Команда"] == "Помаранчеві"].drop(columns=["Команда"]).sort_values(by="Rating 2.0", ascending=False)
            
            format_dict = {'Rating 2.0': '{:.2f}', 'xG': '{:.2f}'}
            
            col_b, col_o = st.columns(2)
            
            with col_b:
                st.markdown(f"### 🔵 Сині ({blue_names})")
                if not blue_team.empty:
                    styled_blue = blue_team.style.highlight_max(subset=['Rating 2.0'], color='#2e7d32').format(format_dict)
                    st.dataframe(styled_blue, width='stretch', hide_index=True)
                else:
                    st.info("Гравців не знайдено")
                    
            with col_o:
                st.markdown(f"### 🟠 Помаранчеві ({orange_names})")
                if not orange_team.empty:
                    styled_orange = orange_team.style.highlight_max(subset=['Rating 2.0'], color='#2e7d32').format(format_dict)
                    st.dataframe(styled_orange, width='stretch', hide_index=True)
                else:
                    st.info("Гравців не знайдено")
            st.divider()

            # ==========================================
            # 2. ПОЗИЦІЮВАННЯ
            # ==========================================
            st.subheader("📍 Позиціювання відносно тімейтів")
            teams = {"Сині": [p for p,c in player_colors.items() if c=='royalblue'], "Помаранчеві": [p for p,c in player_colors.items() if c=='darkorange']}
            
            RATIO = 1.458392
            fig_pos, ax_pos = plt.subplots(figsize=(7, 7 / RATIO))
            fig_pos.patch.set_alpha(0.0) 
            ax_pos.patch.set_alpha(0.0)
            x_max = 4096 * RATIO
            
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread('boostmap.png')
                ax_pos.imshow(img, extent=[-x_max, x_max, -4096, 4096], zorder=1, alpha=0.8)
            except FileNotFoundError: pass
            
            anchors = {'royalblue': {'x': 0, 'y': -2560}, 'darkorange': {'x': 0, 'y': 2560}}
            SCALE_FACTOR = 4.5 
            
            for t_name, members in teams.items():
                if len(members) < 1: continue # Тепер працює для будь-якого режиму гри!
                    
                color = player_colors.get(members[0], 'cyan')
                anchor = anchors.get(color, {'x': 0, 'y': 0})
                
                x_cols, y_cols = [f"{n}_x" for n in members], [f"{n}_y" for n in members]
                t_df = full_df[x_cols + y_cols].dropna(how='all').copy()
                if t_df.empty: continue
                    
                cx_vals, cy_vals = t_df[x_cols].mean(axis=1), t_df[y_cols].mean(axis=1)
                
                player_stats_pos = []
                for name in members:
                    rel_x = (t_df[f"{name}_x"] - cx_vals).mean()
                    rel_y = (t_df[f"{name}_y"] - cy_vals).mean()
                    player_stats_pos.append({'name': name, 'rel_x': rel_x, 'rel_y': rel_y})
                    
                player_stats_pos.sort(key=lambda p: p['name'].lower())
                    
                for i, p_stat in enumerate(player_stats_pos):
                    role_num = i + 1 
                    exaggerated_x, exaggerated_y = p_stat['rel_x'] * SCALE_FACTOR, p_stat['rel_y'] * SCALE_FACTOR
                    draw_x, draw_y = anchor['y'] + exaggerated_y, -(anchor['x'] + exaggerated_x)
                    
                    ax_pos.scatter(draw_x, draw_y, color=color, s=450, edgecolors='white', linewidth=2.5, zorder=5)
                    ax_pos.text(draw_x, draw_y, str(role_num), color='white', ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)
                    bbox_props = dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5)
                    ax_pos.text(draw_x, draw_y + 450, p_stat['name'], color='white', ha='center', fontsize=10, fontweight='bold', bbox=bbox_props, zorder=6)
            
            ax_pos.set_xlim(-x_max, x_max); ax_pos.set_ylim(-4096, 4096); ax_pos.set_aspect('equal'); ax_pos.axis('off')
            fig_pos.tight_layout(pad=0) 
            
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2: st.pyplot(fig_pos, transparent=True, width='content')

            st.divider()

            # ==========================================
            # 3. ШВИДКІСТЬ ГРАВЦІВ
            # ==========================================
            st.subheader("⚡ Швидкість (Average Speed)")
            avg_speeds = {}
            for n in player_paths.keys():
                if f"{n}_speed" in full_df.columns:
                    s_series = full_df[f"{n}_speed"].dropna()
                    if not s_series.empty: avg_speeds[n] = s_series.mean()
                        
            sorted_s = dict(sorted(avg_speeds.items(), key=lambda x: x[1], reverse=True))
            
            fig_spd, ax_spd = plt.subplots(figsize=(10, 4.5))
            fig_spd.patch.set_alpha(0.0) 
            ax_spd.patch.set_alpha(0.0)
            
            bar_colors = ['#3978b5' if player_colors.get(n) == 'royalblue' else '#cc5628' for n in sorted_s.keys()]
            bars = ax_spd.bar(sorted_s.keys(), sorted_s.values(), width=0.96, color=bar_colors, edgecolor='#1a1a1a', linewidth=1)
            
            ax_spd.set_yticks([0, 15, 30, 45, 60])
            ax_spd.yaxis.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax_spd.set_axisbelow(True)
            
            ax_spd.spines['top'].set_visible(False); ax_spd.spines['right'].set_visible(False); ax_spd.spines['left'].set_visible(False) 
            ax_spd.spines['bottom'].set_color('gray')
            ax_spd.tick_params(axis='y', colors='lightgray', length=0, labelsize=10) 
            ax_spd.tick_params(axis='x', colors='white', length=0, labelsize=11)
            ax_spd.bar_label(bars, fmt='%.1f', padding=4, color='white', fontweight='bold', fontsize=11)
            
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2: st.pyplot(fig_spd, transparent=True, width='content')

            st.divider()

            # ==========================================
            # 4. ТЕПЛОВІ КАРТИ
            # ==========================================
            st.subheader("🔥 Теплові карти активності")
            
            # Елементи керування прямо над теплокартою
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                selected_heatmap = st.selectbox("Оберіть гравця або команду:", ["Всі гравці", "Команда Синіх", "Команда Помаранчевих"] + list(player_paths.keys()))
            with col_ctrl2:
                resolution = st.slider("Деталізація (Кількість блоків)", min_value=50, max_value=400, value=150, step=10)
            
            fig_heat, ax_heat = plt.subplots(figsize=(6, 6 / RATIO))
            fig_heat.patch.set_alpha(0.0) 
            ax_heat.patch.set_alpha(0.0)
            
            try:
                img = mpimg.imread('boostmap.png')
                ax_heat.imshow(img, extent=[-x_max, x_max, -4096, 4096], zorder=1)
                ax_heat.imshow(img, extent=[-x_max, x_max, -4096, 4096], zorder=4, alpha=0.3)
            except FileNotFoundError:
                ax_heat.add_patch(plt.Rectangle((-5120, -4096), 10240, 8192, fill=False, color='white', alpha=0.3, linewidth=2))
                
            all_x, all_y = [], []
            if selected_heatmap == "Всі гравці":
                for coords in player_paths.values():
                    all_x.extend(coords['y']); all_y.extend(coords['x'])
            elif selected_heatmap == "Команда Синіх":
                for name, color in player_colors.items():
                    if color == 'royalblue':
                        all_x.extend(player_paths[name]['y']); all_y.extend(player_paths[name]['x'])
            elif selected_heatmap == "Команда Помаранчевих":
                for name, color in player_colors.items():
                    if color == 'darkorange':
                        all_x.extend(player_paths[name]['y']); all_y.extend(player_paths[name]['x'])
            else:
                all_x.extend(player_paths[selected_heatmap]['y']); all_y.extend(player_paths[selected_heatmap]['x'])
                
            if all_x and all_y:
                ax_heat.hist2d(all_x, all_y, bins=(int(resolution * RATIO), resolution), cmap='turbo', norm=mcolors.LogNorm(vmin=1), alpha=0.85, zorder=3)
                
                arr_x = np.array(all_x)
                total_frames = len(arr_x)
                left_pct, mid_pct, right_pct = (arr_x < -1706).sum() / total_frames * 100, ((arr_x >= -1706) & (arr_x <= 1706)).sum() / total_frames * 100, (arr_x > 1706).sum() / total_frames * 100
                
                lbl_left, lbl_mid, lbl_right = "Синя пол.", "Центр", "Пом. пол."
                if selected_heatmap == "Команда Синіх" or player_colors.get(selected_heatmap) == 'royalblue': lbl_left, lbl_right = "Захист", "Атака"
                elif selected_heatmap == "Команда Помаранчевих" or player_colors.get(selected_heatmap) == 'darkorange': lbl_left, lbl_right = "Атака", "Захист"
                    
                ax_heat.axvline(-1706, color='white', linestyle='-', linewidth=2.5, alpha=0.8, zorder=5)
                ax_heat.axvline(1706, color='white', linestyle='-', linewidth=2.5, alpha=0.8, zorder=5)
                
                bbox_props = dict(boxstyle="round,pad=0.4", fc="black", ec="white", alpha=0.8, lw=1.5)
                ax_heat.text(-3413, -3400, f"{lbl_left}\n{left_pct:.1f}%", color="white", ha="center", va="center", fontsize=9, fontweight="bold", bbox=bbox_props, zorder=10)
                ax_heat.text(0, -3400, f"{lbl_mid}\n{mid_pct:.1f}%", color="white", ha="center", va="center", fontsize=9, fontweight="bold", bbox=bbox_props, zorder=10)
                ax_heat.text(3413, -3400, f"{lbl_right}\n{right_pct:.1f}%", color="white", ha="center", va="center", fontsize=9, fontweight="bold", bbox=bbox_props, zorder=10)
            
            ax_heat.set_xlim(-x_max, x_max); ax_heat.set_ylim(-4096, 4096); ax_heat.set_aspect('equal'); ax_heat.axis('off')
            fig_heat.tight_layout(pad=0)
            
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2: st.pyplot(fig_heat, transparent=True, width='content')

            st.divider()

            # ==========================================
            # 5. АНАЛІТИКА БУСТУ ТА FIELD TILT
            # ==========================================
            st.subheader("🚀 Аналітика бусту та Field Tilt")
            
            col_b, col_t = st.columns(2)
            
            with col_b:
                st.markdown("#### 🔋 Аналітика Бусту")
                boost_stats = []
                for name, levels in player_boost_data.items():
                    if levels: 
                        boost_stats.append({
                            "Гравець": name, 
                            "Середній буст": round(np.mean(levels), 1), 
                            "Час на 0%": round((levels.count(0) / len(levels)) * 100, 1), # Повернули розрахунок 0%
                            "Колір": player_colors.get(name, 'cyan')
                        })
                
                if boost_stats:
                    # Створюємо два вертикальні підграфіки (2 рядки, 1 колонка)
                    fig_b, (ax_b1, ax_b2) = plt.subplots(2, 1, figsize=(6, 5.5)) 
                    fig_b.patch.set_alpha(0.0)
                    
                    # 1. Графік середнього бусту
                    df_boost = pd.DataFrame(boost_stats).sort_values("Середній буст", ascending=True)
                    ax_b1.patch.set_alpha(0.0)
                    bars1 = ax_b1.barh(df_boost["Гравець"], df_boost["Середній буст"], color=df_boost["Колір"])
                    ax_b1.bar_label(bars1, padding=3, color='white', fontsize=10, fmt='%.1f')
                    ax_b1.set_xlim(0, 105)
                    ax_b1.set_title("Середній рівень у баку (%)", color='white', pad=10, fontweight='bold')
                    ax_b1.spines['top'].set_visible(False); ax_b1.spines['right'].set_visible(False); ax_b1.spines['left'].set_visible(False)
                    ax_b1.spines['bottom'].set_color('gray')
                    ax_b1.tick_params(colors='white')
                    
                    # 2. Графік часу на 0% бусту (Starvation)
                    df_starve = pd.DataFrame(boost_stats).sort_values("Час на 0%", ascending=True)
                    ax_b2.patch.set_alpha(0.0)
                    bars2 = ax_b2.barh(df_starve["Гравець"], df_starve["Час на 0%"], color='#ff4b4b') # Червоний колір
                    ax_b2.bar_label(bars2, padding=3, color='white', fontsize=10, fmt='%.1f%%')
                    ax_b2.set_xlim(0, max(df_starve["Час на 0%"]) + 10 if not df_starve.empty else 100)
                    ax_b2.set_title("Час без бусту (0%)", color='white', pad=10, fontweight='bold')
                    ax_b2.spines['top'].set_visible(False); ax_b2.spines['right'].set_visible(False); ax_b2.spines['left'].set_visible(False)
                    ax_b2.spines['bottom'].set_color('gray')
                    ax_b2.tick_params(colors='white')
                    
                    fig_b.tight_layout(pad=2.0)
                    st.pyplot(fig_b, transparent=True)
                else:
                    st.warning("Дані про буст не знайдено.")

            with col_t:
                st.markdown("#### 📈 Таймлайн тиску (Field Tilt)")
                if ball_y:
                    y_array = np.array(ball_y)
                    time_axis = np.linspace(0, 300, len(y_array))
                    fig_time, ax_time = plt.subplots(figsize=(6, 4)) # Трохи збільшили висоту для балансу
                    fig_time.patch.set_alpha(0.0); ax_time.patch.set_alpha(0.0)
                    
                    smooth_y = pd.Series(ball_y).rolling(window=300, center=True).mean()
                    ax_time.plot(time_axis, smooth_y, color='white', alpha=0.8, linewidth=1)
                    ax_time.axhline(0, color='gray', linestyle='--', alpha=0.5)
                    ax_time.fill_between(time_axis, smooth_y, 0, where=(smooth_y > 0), color='orange', alpha=0.4)
                    ax_time.fill_between(time_axis, smooth_y, 0, where=(smooth_y < 0), color='royalblue', alpha=0.4)
                    
                    def format_func(x, pos): return f"{int(x // 60):02d}:{int(x % 60):02d}"
                    ax_time.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
                    ax_time.set_xticks(np.arange(0, 301, 60))
                    ax_time.set_xlim(0, 300)
                    ax_time.spines['top'].set_visible(False); ax_time.spines['right'].set_visible(False)
                    ax_time.spines['bottom'].set_color('gray'); ax_time.spines['left'].set_color('gray')
                    ax_time.tick_params(colors='white')
                    st.pyplot(fig_time, transparent=True)

            st.divider()
            
            # ==========================================
            # 6. КАРТА УДАРІВ / xG
            # ==========================================
            st.subheader("🎯 Карта небезпечних моментів (xG > 0.15)")
            if not hits_df.empty and len(hits_df[hits_df['xG'] > 0.15]) > 0:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig_shots, ax_shots = plt.subplots(figsize=(5, 5 * 1.458))
                    fig_shots.patch.set_alpha(0.0); ax_shots.patch.set_alpha(0.0)
                    try:
                        img = mpimg.imread('boostmap.png')
                        img_rotated = np.rot90(img, k=1)
                        ax_shots.imshow(img_rotated, extent=[-4096, 4096, -5972, 5972], zorder=1, alpha=0.4)
                    except: pass
                    
                    for _, row in hits_df[hits_df['xG'] > 0.15].iterrows():
                        ax_shots.scatter(row['Ball_X'], row['Ball_Y'], s=row['xG']*400, color=player_colors.get(row['Player'], 'white'), alpha=0.9, zorder=5, edgecolors='black')
                    
                    ax_shots.set_xlim(-4096, 4096); ax_shots.set_ylim(-5972, 5972); ax_shots.set_aspect('equal'); ax_shots.axis('off')
                    fig_shots.tight_layout(pad=0)
                    st.pyplot(fig_shots, transparent=True)
            else:
                st.info("В цьому матчі не було зафіксовано небезпечних моментів.")

            st.divider()

            # ==========================================
            # 7. AI COACH (ШІ-ТРЕНЕР V2)
            # ==========================================
            st.subheader("🧠 AI Coach (Тактичний Аналіз)")
            
            if 'AI_Score' in hits_df.columns:
                
                # --- АНТИ-КІКОФ ФІЛЬТР (ПРО) ---
                # Збільшуємо зону ігнорування до 1500 юнітів (15 метрів). 
                # Це розмір центрального кола. Ігноруємо все місиво одразу після старту.
                active_play_df = hits_df[~((hits_df['Ball_X'].abs() < 1500) & (hits_df['Ball_Y'].abs() < 1500))]

                # Тепер шукаємо помилки тільки в АКТИВНІЙ грі (без кікофів)
                bad_plays = active_play_df[active_play_df['AI_Score'] < 25].sort_values('AI_Score')
                
                # РОЗУМНИЙ ФІЛЬТР ДЛЯ ХАЙЛАЙТІВ (Відсікаємо моменти, де немає суперників)
                good_plays = active_play_df[
                    (active_play_df['AI_Score'] > 75) & 
                    (active_play_df['Nearest_Opponent'] < 1800) &  # Має бути тиск від суперника
                    (active_play_df['Nearest_Teammate'] > 500)     # Це не має бути випадковий дабл-коміт
                ].sort_values('AI_Score', ascending=False)
                
                col_ai1, col_ai2 = st.columns(2)
                
                with col_ai1:
                    st.markdown("#### 🚨 Критичні помилки")
                    st.caption("Дабл-коміти, пустий буст або зламана ротація.")
                    if not bad_plays.empty:
                        for i, row in bad_plays.head(5).iterrows():
                            time_sec = int(row['Frame'] / 30)
                            time_str = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
                            
                            # --- ГЕНЕРАТОР ПОЯСНЕНЬ (ЧОМУ ПОМИЛКА?) ---
                            reasons = []
                            if row['Nearest_Teammate'] < 1200: reasons.append("⚠️ Дабл-коміт (зрізав тімейта)")
                            if row['Boost_Amount'] < 15: reasons.append("⚠️ Пішов на м'яч без бусту")
                            if row['Player_Speed'] < 400: reasons.append("⚠️ Стояв укопаний (швидкість ~0)")
                            if row['Is_Last_Man'] == 1: reasons.append("⚠️ Ризикував останнім у захисті")
                            if not reasons: reasons.append("⚠️ Просто програв позицію (поганий таймінг)")
                            
                            reason_txt = " • ".join(reasons)
                            
                            st.error(f"⏱ **{time_str}** | **{row['Player']}** | Шанс: **{row['AI_Score']:.1f}%**\n\n**Чому:** {reason_txt}\n\n🔋 Буст: {int(row['Boost_Amount'])}% | 🏎️ Швидкість: {int(row['Player_Speed'])}")
                    else:
                        st.info("ШІ не знайшов критичних помилок!")

                with col_ai2:
                    st.markdown("#### ✅ Геніальні рішення")
                    st.caption("Виграні челенджі під тиском та ідеальний таймінг.")
                    if not good_plays.empty:
                        for i, row in good_plays.head(5).iterrows():
                            time_sec = int(row['Frame'] / 30)
                            time_str = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
                            
                            # --- ГЕНЕРАТОР ПОЯСНЕНЬ (ЧОМУ УСПІХ?) ---
                            reasons = []
                            if row['Nearest_Opponent'] < 800: reasons.append("🔥 Виграв жорсткий 50/50")
                            elif row['Nearest_Opponent'] < 1800: reasons.append("✅ Обіграв суперника під тиском")
                            if row['Boost_Amount'] > 50: reasons.append("✅ Зберіг багато ресурсів")
                            if row['Player_Speed'] > 1500: reasons.append("✅ Ідеальний імпульс (Supersonic)")
                            if not reasons: reasons.append("✅ Дуже надійна макро-гра")
                            
                            reason_txt = " • ".join(reasons)
                            
                            st.success(f"⏱ **{time_str}** | **{row['Player']}** | Шанс: **{row['AI_Score']:.1f}%**\n\n**Чому:** {reason_txt}\n\n🔋 Буст: {int(row['Boost_Amount'])}% | 🏎️ Швидкість: {int(row['Player_Speed'])}")
                    else:
                        st.info("Не знайдено геніальних дій.")
            else:
                st.warning("⚠️ Файл моделі 'ai_coach_model_v2.pkl' не знайдено.")

            # ==========================================
            # 8. ТАКТИЧНА ДОШКА (FREEZE-FRAME)
            # ==========================================
            st.divider()
            st.subheader("🗺️ Тактична дошка (Freeze-Frame)")
            st.caption("Позиції всіх гравців у момент дотику. Детальний візуальний та текстовий розбір від ШІ.")

            analyze_options = []
            
            # Збираємо всі цікаві моменти в один список для вибору
            if 'AI_Score' in hits_df.columns:
                if 'bad_plays' in locals() and not bad_plays.empty:
                    for idx, row in bad_plays.head(5).iterrows():
                        time_sec = int(row['Frame'] / 30)
                        time_str = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
                        analyze_options.append(f"🚨 Помилка | {time_str} | {row['Player']} | Frame: {row['Frame']}")
                        
                if 'good_plays' in locals() and not good_plays.empty:
                    for idx, row in good_plays.head(5).iterrows():
                        time_sec = int(row['Frame'] / 30)
                        time_str = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
                        analyze_options.append(f"✅ Успіх | {time_str} | {row['Player']} | Frame: {row['Frame']}")

            if analyze_options:
                selected_moment = st.selectbox("Оберіть момент для розбору:", analyze_options)
                
                # Дістаємо кадр та ім'я гравця з обраного тексту
                target_frame = int(selected_moment.split("Frame: ")[1])
                target_player = selected_moment.split(" | ")[2]
                
                # --- НОВЕ: РОЗДІЛЯЄМО НА 2 КОЛОНКИ (ТЕКСТ + КАРТА) ---
                col_text, col_plot = st.columns([1, 2])
                
                with col_text:
                    moment_data = hits_df[(hits_df['Frame'] == target_frame) & (hits_df['Player'] == target_player)]
                    if not moment_data.empty:
                        row = moment_data.iloc[0]
                        score = row['AI_Score']
                        
                        st.markdown(f"### 🤖 Розбір від ШІ")
                        st.metric("Шанс успіху", f"{score:.1f}%")
                        
                        st.markdown("#### Що зафіксували сенсори:")
                        st.write(f"🔋 **Буст:** {int(row['Boost_Amount'])}%")
                        st.write(f"🏎️ **Швидкість:** {int(row['Player_Speed'])}")
                        # ДІЛИМО НА 100 ДЛЯ ПЕРЕВОДУ В МЕТРИ
                        st.write(f"👥 **Тімейт:** за {row['Nearest_Teammate'] / 100:.1f} м")
                        st.write(f"⚔️ **Суперник:** за {row['Nearest_Opponent'] / 100:.1f} м")
                        
                        st.markdown("#### Вердикт:")
                        reasons = []
                        if score < 50: # ПОМИЛКА
                            if row['Nearest_Teammate'] < 1200: reasons.append("⚠️ **Дабл-коміт:** Подивись на лінію до тімейта. Ви стрибали в один м'яч і заважали один одному.")
                            if row['Boost_Amount'] < 15: reasons.append("⚠️ **Без ресурсів:** Ти поліз на цей м'яч майже з пустим баком бусту.")
                            if row['Player_Speed'] < 400: reasons.append("⚠️ **Втрата імпульсу:** Ти стояв укопаний. Без швидкості неможливо виграти челендж.")
                            if row['Is_Last_Man'] == 1: reasons.append("⚠️ **Ризик останнім:** Ти поліз на м'яч, будучи останнім гравцем у захисті. Це відкрило пусті ворота.")
                            if not reasons: reasons.append("⚠️ **Погана позиція:** Суперник мав кращий кут, і ти не мав шансів його перебити.")
                            
                            for r in reasons: st.error(r)
                            
                        else: # УСПІХ
                            if row['Nearest_Opponent'] < 800: reasons.append("🔥 **Жорсткий 50/50:** Суперник був впритул, але ти ідеально зіграв у корпус і виграв м'яч.")
                            elif row['Nearest_Opponent'] < 1800: reasons.append("✅ **Крутий аутплей:** Ти обіграв суперника, який намагався тебе пресингувати.")
                            if row['Boost_Amount'] > 50: reasons.append("✅ **Грамотний менеджмент:** Ти зробив дотик і зберіг багато бусту для наступної дії.")
                            if row['Player_Speed'] > 1500: reasons.append("✅ **Збереження імпульсу:** Ти летів на швидкості Supersonic і зберіг ротацію.")
                            if not reasons: reasons.append("✅ **Надійна гра:** Все зроблено за підручником.")
                            
                            for r in reasons: st.success(r)

                with col_plot:
                    fig_board, ax_board = plt.subplots(figsize=(6, 6 * 1.458))
                    fig_board.patch.set_alpha(0.0)
                    ax_board.patch.set_alpha(0.0)
                    
                    try:
                        img = mpimg.imread('boostmap.png')
                        img_rotated = np.rot90(img, k=1)
                        ax_board.imshow(img_rotated, extent=[-4096, 4096, -5972, 5972], zorder=1, alpha=0.5)
                    except:
                        ax_board.add_patch(plt.Rectangle((-4096, -5120), 8192, 10240, fill=False, color='white', alpha=0.3))

                    if target_frame in full_df.index:
                        # 1. М'яч
                        bx = full_df.loc[target_frame, "ball_x"]
                        by = full_df.loc[target_frame, "ball_y"]
                        if pd.notna(bx) and pd.notna(by):
                            ax_board.scatter(bx, by, color='white', s=150, edgecolors='black', linewidth=2, zorder=10, label="М'яч")
                            
                        # 2. Гравці
                        target_px, target_py = None, None
                        target_color = player_colors.get(target_player, 'cyan')
                        
                        for p_name, color in player_colors.items():
                            if f"{p_name}_x" in full_df.columns:
                                px = full_df.loc[target_frame, f"{p_name}_x"]
                                py = full_df.loc[target_frame, f"{p_name}_y"]
                                
                                if pd.notna(px) and pd.notna(py):
                                    is_target = (p_name == target_player)
                                    size = 450 if is_target else 150
                                    edge_color = '#ffeb3b' if is_target else 'white'
                                    lw = 3 if is_target else 1
                                    alpha_val = 1.0 if is_target else 0.7
                                    
                                    ax_board.scatter(px, py, color=color, s=size, edgecolors=edge_color, linewidth=lw, alpha=alpha_val, zorder=8)
                                    ax_board.text(px, py + 350, p_name, color='white', ha='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, ec="none"), zorder=9)
                                    
                                    if is_target:
                                        target_px, target_py = px, py
                                        
                        # 3. Радар Дабл-Комітів та Тиску (У МЕТРАХ)
                        if target_px is not None and target_py is not None:
                            for p_name, color in player_colors.items():
                                if p_name == target_player: continue
                                if f"{p_name}_x" in full_df.columns:
                                    px = full_df.loc[target_frame, f"{p_name}_x"]
                                    py = full_df.loc[target_frame, f"{p_name}_y"]
                                    
                                    if pd.notna(px) and pd.notna(py):
                                        dist = math.sqrt((px - target_px)**2 + (py - target_py)**2)
                                        
                                        if color == target_color and dist < 1500:
                                            ax_board.plot([target_px, px], [target_py, py], color='#ff4b4b', linestyle='--', linewidth=2.5, zorder=7)
                                            # Замінили одиниці на метри
                                            ax_board.text((target_px + px)/2, (target_py + py)/2, f"Дабл-коміт!\n({dist / 100:.1f} м)", color='white', fontsize=8, fontweight='bold', ha='center', bbox=dict(fc="#ff4b4b", alpha=0.8, ec="none"), zorder=11)
                                            
                                        elif color != target_color and dist < 1800:
                                            ax_board.plot([target_px, px], [target_py, py], color='orange', linestyle=':', linewidth=2, zorder=7)
                                            # Замінили одиниці на метри
                                            ax_board.text((target_px + px)/2, (target_py + py)/2, f"Тиск\n({dist / 100:.1f} м)", color='white', fontsize=7, fontweight='bold', ha='center', bbox=dict(fc="orange", alpha=0.7, ec="none"), zorder=11)

                    ax_board.set_xlim(-4096, 4096)
                    ax_board.set_ylim(-5972, 5972)
                    ax_board.set_aspect('equal')
                    ax_board.axis('off')
                    
                    st.pyplot(fig_board, transparent=True, width='stretch')

else:
    st.info("Будь ласка, завантажте файл .replay у бічній панелі для генерації повного дашборду.")