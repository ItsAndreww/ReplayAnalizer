import platform
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
    
    if platform.system() == "Windows":
        exe_name = "rrrocket.exe"
    else:
        exe_name = "rrrocket"
        
    exe_path = os.path.join(os.getcwd(), exe_name) 
    
    if platform.system() != "Windows" and os.path.exists(exe_path):
        os.chmod(exe_path, 0o755)
    
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
        st.error(f"Помилка парсера: {result.stderr}")
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
    
    is_overtime = False
    if properties.get('bOverTime'): is_overtime = True
    if properties.get('NumOvertimes', 0) > 0: is_overtime = True
    if data.get('gameMetadata', {}).get('isOvertime'): is_overtime = True
        
    known_names = []
    raw_stats = properties.get('PlayerStats', [])
    if not isinstance(raw_stats, list): raw_stats = [raw_stats]
    
    player_true_team = {}  # Створюємо словник для точних команд
    
    for player in raw_stats:
        if isinstance(player, dict):
            p_name = player.get('Name', 'Unknown')
            known_names.append(p_name)
            
            # Витягуємо точний індекс команди (0 - Сині, 1 - Помаранчеві)
            if 'Team' in player:
                player_true_team[p_name] = player['Team']
                
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
    
    # --- ФІКС: ТЕПЕР МИ ЗБИРАЄМО ВИСОТУ (Z) ТА ВЕКТОРИ (VZ) ---
    synced_data = {name: {'x': {}, 'y': {}, 'z': {}, 'speed': {}, 'vx': {}, 'vy': {}, 'vz': {}, 'boost': {}} for name in known_names}
    ball_synced = {'x': {}, 'y': {}, 'z': {}, 'vx': {}, 'vy': {}, 'vz': {}}

    def get_exact_obj_id(obj_name):
        for i, o in enumerate(objects):
            if obj_name in o and "Default__" not in o:
                return i
        return None

    pri_obj_id = get_exact_obj_id("Pawn:PlayerReplicationInfo")
    team_obj_id = get_exact_obj_id("PlayerReplicationInfo:Team")
    vehicle_obj_id = get_exact_obj_id("CarComponent_TA:Vehicle")
    
    overtime_obj_id = None
    overtime_start_frame = None
    for i, o in enumerate(objects):
        if "bOverTime" in o:
            overtime_obj_id = i
            break

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

                if object_id == overtime_obj_id:
                    if isinstance(attribute, dict) and attribute.get('Boolean') is True:
                        is_overtime = True
                        if overtime_start_frame is None:
                            overtime_start_frame = frame_idx

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
                        synced_data[p_name]['boost'][frame_idx] = boost_pct

                if 'RigidBody' in attribute and attribute['RigidBody']:
                    rb = attribute['RigidBody']
                    loc, vel = rb.get('location') or {}, rb.get('linear_velocity') or {}
                    
                    if actor_id == ball_actor_id:
                        if 'x' in loc: ball_synced['x'][frame_idx] = loc['x']
                        if 'y' in loc: 
                            ball_y_timeline.append(loc['y'])
                            ball_synced['y'][frame_idx] = loc['y']
                        if 'z' in loc: ball_synced['z'][frame_idx] = loc['z'] # ЗБІР Z
                        
                        if 'x' in vel: ball_synced['vx'][frame_idx] = vel['x']
                        if 'y' in vel: ball_synced['vy'][frame_idx] = vel['y']
                        if 'z' in vel: ball_synced['vz'][frame_idx] = vel['z'] # ЗБІР VZ
                        
                    pid = car_to_pri.get(actor_id)
                    if pid and pid in pri_to_name:
                        pname = pri_to_name[pid]
                        if pname in known_names:
                            if 'x' in loc and 'y' in loc:
                                player_paths[pname]['x'].append(loc['x'])
                                player_paths[pname]['y'].append(loc['y'])
                                synced_data[pname]['x'][frame_idx] = loc['x']
                                synced_data[pname]['y'][frame_idx] = loc['y']
                            if 'z' in loc:
                                synced_data[pname]['z'][frame_idx] = loc['z'] # ЗБІР Z гравця
                                
                            if 'x' in vel and 'y' in vel and 'z' in vel:
                                sp = ((vel['x']**2 + vel['y']**2 + vel['z']**2)**0.5) * 0.036
                                synced_data[pname]['speed'][frame_idx] = sp
                                synced_data[pname]['vx'][frame_idx] = vel['x']
                                synced_data[pname]['vy'][frame_idx] = vel['y']
                                synced_data[pname]['vz'][frame_idx] = vel['z'] 

    full_df = pd.DataFrame(index=range(len(frames)))
    if len(frames) > 0:
        for name in known_names:
            full_df[f"{name}_x"] = pd.to_numeric(pd.Series(synced_data[name]['x']).reindex(full_df.index).ffill(), errors='coerce')
            full_df[f"{name}_y"] = pd.to_numeric(pd.Series(synced_data[name]['y']).reindex(full_df.index).ffill(), errors='coerce')
            full_df[f"{name}_z"] = pd.to_numeric(pd.Series(synced_data[name]['z']).reindex(full_df.index).ffill(), errors='coerce') # КОЛОНКА Z
            full_df[f"{name}_speed"] = pd.to_numeric(pd.Series(synced_data[name]['speed']).reindex(full_df.index).ffill(limit=120).fillna(0), errors='coerce')
            full_df[f"{name}_vx"] = pd.to_numeric(pd.Series(synced_data[name]['vx']).reindex(full_df.index).ffill().fillna(0), errors='coerce')
            full_df[f"{name}_vy"] = pd.to_numeric(pd.Series(synced_data[name]['vy']).reindex(full_df.index).ffill().fillna(0), errors='coerce')
            full_df[f"{name}_vz"] = pd.to_numeric(pd.Series(synced_data[name]['vz']).reindex(full_df.index).ffill().fillna(0), errors='coerce') # КОЛОНКА VZ
            full_df[f"{name}_boost"] = pd.to_numeric(pd.Series(synced_data[name]['boost']).reindex(full_df.index).ffill().fillna(33), errors='coerce')
            
        full_df["ball_x"] = pd.to_numeric(pd.Series(ball_synced['x']).reindex(full_df.index).ffill(), errors='coerce')
        full_df["ball_y"] = pd.to_numeric(pd.Series(ball_synced['y']).reindex(full_df.index).ffill(), errors='coerce')
        full_df["ball_z"] = pd.to_numeric(pd.Series(ball_synced['z']).reindex(full_df.index).ffill(), errors='coerce') # КОЛОНКА М'ЯЧА Z
        full_df["ball_vx"] = pd.to_numeric(pd.Series(ball_synced['vx']).reindex(full_df.index).ffill(), errors='coerce')
        full_df["ball_vy"] = pd.to_numeric(pd.Series(ball_synced['vy']).reindex(full_df.index).ffill(), errors='coerce')
        full_df["ball_vz"] = pd.to_numeric(pd.Series(ball_synced['vz']).reindex(full_df.index).ffill(), errors='coerce')

    hits_data = []
    if not full_df.empty:
        for name in known_names:
            # 3D ДИСТАНЦІЯ ДО М'ЯЧА (Тепер бере до уваги висоту!)
            dist_to_ball = np.sqrt((full_df[f"{name}_x"] - full_df["ball_x"])**2 + 
                                   (full_df[f"{name}_y"] - full_df["ball_y"])**2 + 
                                   (full_df[f"{name}_z"] - full_df["ball_z"])**2)
            hit_frames = full_df[dist_to_ball < 300].index.tolist()
            last_hit_frame = -999
            for frame in hit_frames:
                if frame - last_hit_frame > 30:
                    hits_data.append({'Player': name, 'Ball_X': full_df.loc[frame, "ball_x"], 'Ball_Y': full_df.loc[frame, "ball_y"], 'Frame': frame})
                    last_hit_frame = frame
    hits_df = pd.DataFrame(hits_data)

    unique_teams = sorted([t for t in set(pri_to_team.values()) if t is not None])
    p_colors = {name: 'royalblue' for name in known_names}
    for pri_id, p_name in pri_to_name.items():
        t_id = pri_to_team.get(pri_id)
        if p_name in p_colors and len(unique_teams) >= 2 and t_id == unique_teams[1]:
            p_colors[p_name] = 'darkorange'

    ot_duration_secs = 0
    if is_overtime and overtime_start_frame is not None and not hits_df.empty:
        final_hit_frame = hits_df['Frame'].max()
        if final_hit_frame > overtime_start_frame:
            ot_duration_secs = (final_hit_frame - overtime_start_frame) // 30

    match_info = {
        'is_overtime': is_overtime,
        'ot_duration_secs': ot_duration_secs,
        'total_replay_secs': len(frames) // 30
    }

    return player_paths, p_colors, match_stats, ball_y_timeline, player_boost_data, full_df, hits_df, match_info

def calculate_xg(x, y, team_color):
    target_y = 5120 if team_color == 'royalblue' else -5120
    if (team_color == 'royalblue' and y < 0) or (team_color == 'darkorange' and y > 0):
        return 0.0
    distance = math.sqrt(x**2 + (target_y - y)**2)
    dist_factor = max(0.0, 1.0 - (distance / 4500)**1.5)
    angle_factor = max(0.0, 1.0 - (abs(x) / 3500)**2)
    base_xg = dist_factor * angle_factor
    return max(0.01, min(0.95, base_xg))


import matplotlib.animation as animation
import tempfile

def generate_moment_gif(full_df, player_colors, target_frame, known_names, fps=15):
    """Генерує GIF анімацію ±3 секунди навколо моменту"""
    
    TRAIL_FRAMES = 8
    start_frame = max(0, target_frame - 90)
    end_frame = min(len(full_df) - 1, target_frame + 90)
    frames_to_render = list(range(start_frame, end_frame + 1, 2))  # кожен 2-й фрейм = плавніше
    
    RATIO = 1.458
    fig, ax = plt.subplots(figsize=(4, 4 * RATIO))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Завантаж фон один раз
    bg_img = None
    try:
        import matplotlib.image as mpimg
        bg_img = mpimg.imread('boostmap.png')
        bg_img_rotated = np.rot90(bg_img, k=3)
    except:
        bg_img_rotated = None
    
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.transforms as mtransforms

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor('#1a1a2e')
        
        # Фон
        if bg_img_rotated is not None:
            ax.imshow(bg_img_rotated, extent=[-4096, 4096, -5972, 5972], zorder=1, alpha=0.3)
        else:
            ax.add_patch(plt.Rectangle((-4096, -5120), 8192, 10240,
                        fill=False, color='white', alpha=0.2, zorder=2))
            ax.add_patch(plt.Rectangle((-893, 5120), 1786, 100,
                        fill=True, color='orange', alpha=0.5, zorder=3))
            ax.add_patch(plt.Rectangle((-893, -5220), 1786, 100,
                        fill=True, color='royalblue', alpha=0.5, zorder=3))

        # Бусти
        for bl in [(-3072,-4096),(3072,-4096),(-3584,0),(3584,0),(-3072,4096),(3072,4096)]:
            ax.scatter(bl[0], bl[1], color='#ffff00', s=80, alpha=0.4, edgecolors='none', zorder=3)

        # Виділення моменту
        if frame_idx == target_frame:
            ax.add_patch(plt.Circle((0, 0), 6000, fill=False,
                        color='yellow', alpha=0.15, linewidth=3, zorder=2))

        trail_start = max(0, frame_idx - TRAIL_FRAMES)
        trail_indices = list(range(trail_start, frame_idx + 1))

        for p_name, color in player_colors.items():
            if f"{p_name}_x" not in full_df.columns:
                continue

            px = full_df.loc[frame_idx, f"{p_name}_x"] if frame_idx in full_df.index else None
            py = full_df.loc[frame_idx, f"{p_name}_y"] if frame_idx in full_df.index else None

            if pd.isna(px) or pd.isna(py):
                continue

            pvx = full_df.loc[frame_idx, f"{p_name}_vx"]
            pvy = full_df.loc[frame_idx, f"{p_name}_vy"]

            # --- ТРЕЙЛ ---
            trail_px = full_df.loc[trail_indices, f"{p_name}_x"].dropna().values
            trail_py = full_df.loc[trail_indices, f"{p_name}_y"].dropna().values
            if len(trail_px) > 1:
                n = len(trail_px)
                for i in range(1, n):
                    ax.plot(trail_px[i-1:i+1], trail_py[i-1:i+1],
                        color=color, alpha=(i/n)*0.6,
                        linewidth=1 + (i/n)*2, zorder=6)

            # --- КУТ ПОВОРОТУ (за вектором швидкості) ---
            if pd.notna(pvx) and pd.notna(pvy) and (abs(pvx) > 100 or abs(pvy) > 100):
                # atan2(y, x) — стандартний кут у matplotlib
                # pvx = рух по X полю, pvy = рух по Y полю
                angle_rad = math.atan2(pvy, pvx) - math.pi / 2
            else:
                # Якщо стоїть — дивиться до свого голу
                color = player_colors.get(p_name, 'royalblue')
                if color == 'royalblue':
                    angle_rad = math.pi  # дивиться вниз (до свого голу -Y)
                else:
                    angle_rad = 0.0      # дивиться вгору (до свого голу +Y)

            # --- РОЗМІР ТРИКУТНИКА ---
            SIZE = 520  # розмір у одиницях поля
            is_target = (p_name == target_player)
            size = SIZE * 1.35 if is_target else SIZE

            # Вершини трикутника
            # ніс = вперед, база = ззаду, вужчий для кращого вигляду
            tip   = np.array([ 0,        size * 0.7])   # гострий ніс
            left  = np.array([-size * 0.38, -size * 0.45])  # лівий задній кут
            right = np.array([ size * 0.38, -size * 0.45])  # правий задній кут

            # Матриця повороту
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rot = np.array([[cos_a, -sin_a],
                            [sin_a,  cos_a]])

            pts = np.array([tip, left, right])
            rotated = (rot @ pts.T).T + np.array([px, py])

            # Основний трикутник
            triangle = plt.Polygon(
                rotated,
                closed=True,
                facecolor=color,
                edgecolor='#ffeb3b' if is_target else 'white',
                linewidth=3.0 if is_target else 1.5,
                zorder=10
            )
            ax.add_patch(triangle)

            # Буква гравця всередині
            center = rotated.mean(axis=0)
            ax.text(center[0], center[1], p_name[0].upper(),
                color='white', ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=11)

            # Підпис імені
            direction = -1 if py > 0 else 1
            va_align = 'top' if direction < 0 else 'bottom'
            ax.text(px, py + (direction * 420), p_name,
                color='white', ha='center', va=va_align,
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.15", fc="black",
                            alpha=0.65, ec="none"), zorder=12)

            # Вектор швидкості (тільки якщо швидкий)
            if pd.notna(pvx) and pd.notna(pvy) and (abs(pvx) > 400 or abs(pvy) > 400):
                ax.quiver(px, py, pvx, pvy,
                        angles='xy', scale_units='xy',
                        scale=10, color=color,
                        width=0.012, headwidth=4, headlength=5,
                        alpha=0.6, zorder=9)

        # --- М'ЯЧ + ТРЕЙЛ ---
        if "ball_x" in full_df.columns:
            trail_bx = full_df.loc[trail_indices, "ball_x"].dropna().values
            trail_by = full_df.loc[trail_indices, "ball_y"].dropna().values
            if len(trail_bx) > 1:
                n = len(trail_bx)
                for i in range(1, n):
                    ax.plot(trail_bx[i-1:i+1], trail_by[i-1:i+1],
                        color='white', alpha=(i/n)*0.5,
                        linewidth=1 + (i/n)*2.5, zorder=7)

            bx = full_df.loc[frame_idx, "ball_x"] if frame_idx in full_df.index else None
            by = full_df.loc[frame_idx, "ball_y"] if frame_idx in full_df.index else None

            if pd.notna(bx) and pd.notna(by):
                bz = full_df.loc[frame_idx, "ball_z"] if "ball_z" in full_df.columns else 0
                bz = bz if pd.notna(bz) else 0
                ball_size = 150 + max(0, (bz / 2000) * 200)

                ax.scatter(bx, by, color='white', s=ball_size,
                        edgecolors='black', linewidth=2, zorder=12)

                bvx = full_df.loc[frame_idx, "ball_vx"]
                bvy = full_df.loc[frame_idx, "ball_vy"]
                if pd.notna(bvx) and pd.notna(bvy) and (abs(bvx) > 100 or abs(bvy) > 100):
                    ax.quiver(bx, by, bvx, bvy,
                            angles='xy', scale_units='xy',
                            scale=9, color='white', width=0.012,
                            headwidth=4, headlength=5, zorder=13)

        # --- ТАЙМЕР ---
        secs_from_target = (frame_idx - target_frame) / 30
        time_color = 'yellow' if frame_idx == target_frame else 'white'
        time_label = "◉ МОМЕНТ" if frame_idx == target_frame else f"{secs_from_target:+.1f}s"
        ax.text(0, -5600, time_label, color=time_color,
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(fc='#1a1a2e', alpha=0.8,
                        boxstyle="round,pad=0.3"), zorder=15)

        ax.set_xlim(-4096, 4096)
        ax.set_ylim(-5972, 5972)
        ax.set_aspect('equal')
        ax.axis('off')
    
    
    # Рендер анімації
    anim = animation.FuncAnimation(
        fig, draw_frame, 
        frames=frames_to_render,
        interval=1000/fps
    )
    
    # Зберігаємо у тимчасовий файл
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
        tmp_path = tmp.name
    
    anim.save(tmp_path, writer='pillow', fps=fps, dpi=80)
    plt.close(fig)
    
    return tmp_path
# --- 2. ГОЛОВНИЙ ІНТЕРФЕЙС ---

st.title("RL Analytics Dashboard")

st.sidebar.header("📁 Завантаження")
uploaded_file = st.sidebar.file_uploader("Оберіть Replay файл", type=['replay'])

if uploaded_file:
    with st.spinner('Автоматичний парсинг реплею та генерація аналітики...'):
        json_data = parse_uploaded_replay(uploaded_file)
        
        if json_data:
            player_paths, player_colors, match_stats, ball_y, player_boost_data, full_df, hits_df, match_info = process_match_data(json_data)
            
            # --- ПЛАШКА ЧАСУ ТА ОВЕРТАЙМУ ---
            is_ot = match_info.get('is_overtime', False)
            ot_secs = match_info.get('ot_duration_secs', 0)
            total_secs = match_info.get('total_replay_secs', 0)
            
            tot_m, tot_s = divmod(total_secs, 60)
            
            if is_ot:
                if ot_secs > 0:
                    ot_m, ot_s = divmod(ot_secs, 60)
                    st.warning(f"⏳ **ОВЕРТАЙМ!** Основний час матчу: **5:00** | Овертайм тривав: **{ot_m}:{ot_s:02d}** (Загальний запис: {tot_m}:{tot_s:02d})")
                else:
                    st.warning(f"⏳ **ОВЕРТАЙМ!** Матч вийшов за межі основного часу. (Загальний запис: {tot_m}:{tot_s:02d})")
            else:
                st.success(f"⏱️ **ОСНОВНИЙ ЧАС.** Матч завершився без овертайму. (Загальний запис: {tot_m}:{tot_s:02d})")
            
            # --- РОЗРАХУНОК xG ТА AI COACH (V3 PRO) ---
            import joblib
            try:
                ai_coach = joblib.load('ai_coach_model_v4.pkl') # <--- ЗАВАНТАЖУЄМО V4
                AI_VERSION = 4
            except:
                try:
                    ai_coach = joblib.load('ai_coach_model_v3.pkl')
                    AI_VERSION = 3
                except:
                    ai_coach = None
                    AI_VERSION = 0

            player_all_touches_xg = {}
            if not hits_df.empty:
                # 1. Ініціалізуємо всі потрібні колонки для V3
                v3_features = [
                    'Ball_X', 'Ball_Y', 'Ball_Z',
                    'Player_X', 'Player_Y', 'Player_Z',
                    'Dist_to_Own_Goal', 'Dist_to_Opp_Goal',
                    'Nearest_Teammate', 'Nearest_Opponent', 'Opp_Dist_To_Ball',
                    'Teammates_Behind_Ball', 'Is_Last_Man',
                    'Boost_Amount', 'Player_Speed', 'Ball_Speed',
                    'Player_VY', 'Player_VX'
                ]
                for col in v3_features:
                    if col not in hits_df.columns:
                        hits_df[col] = 0.0

                for idx, row in hits_df.iterrows():
                    p_name = row['Player']
                    p_color = player_colors.get(p_name, 'cyan')
                    hit_xg = calculate_xg(row['Ball_X'], row['Ball_Y'], p_color)
                    hits_df.at[idx, 'xG'] = hit_xg

                    # Фільтр — тільки реальні шоти йдуть в xG
                    curr_ball_y = row['Ball_Y']
                    curr_ball_x = row['Ball_X']
                    frame = row['Frame']

                    is_shot = False

                    # 1. М'яч у чужій половині
                    in_attack = (p_color == 'royalblue' and curr_ball_y > 1000) or \
                                (p_color == 'darkorange' and curr_ball_y < -1000)
                    
                    # 2. М'яч летить до воріт суперника
                    bvy = full_df.loc[frame, 'ball_vy'] if frame in full_df.index else 0
                    going_to_goal = (p_color == 'royalblue' and bvy > 200) or \
                                    (p_color == 'darkorange' and bvy < -200)
                    
                    # 3. Достатня швидкість м'яча
                    bvx_val = full_df.loc[frame, 'ball_vx'] if frame in full_df.index else 0
                    bvy_val = full_df.loc[frame, 'ball_vy'] if frame in full_df.index else 0
                    bvz_val = full_df.loc[frame, 'ball_vz'] if frame in full_df.index else 0
                    ball_speed = math.sqrt(bvx_val**2 + bvy_val**2 + bvz_val**2)
                    strong_enough = ball_speed > 1200

                    if in_attack and going_to_goal and strong_enough:
                        is_shot = True

                    if is_shot:
                        if p_name not in player_all_touches_xg:
                            player_all_touches_xg[p_name] = []
                        player_all_touches_xg[p_name].append(hit_xg)

                    frame = row['Frame']
                    
                    if f"{p_name}_x" in full_df.columns:
                        # Дістаємо координати з full_df
                        px = full_df.loc[frame, f"{p_name}_x"]
                        py = full_df.loc[frame, f"{p_name}_y"]
                        pz = full_df.loc[frame, f"{p_name}_z"]
                        bx = full_df.loc[frame, "ball_x"]
                        by = full_df.loc[frame, "ball_y"]
                        bz = full_df.loc[frame, "ball_z"]

                        # Записуємо базові координати
                        hits_df.at[idx, 'Ball_X'] = bx
                        hits_df.at[idx, 'Ball_Y'] = by
                        hits_df.at[idx, 'Ball_Z'] = bz
                        hits_df.at[idx, 'Player_X'] = px
                        hits_df.at[idx, 'Player_Y'] = py
                        hits_df.at[idx, 'Player_Z'] = pz
                        
                        # Швидкості
                        pvx = full_df.loc[frame, f"{p_name}_vx"]
                        pvy = full_df.loc[frame, f"{p_name}_vy"]
                        pvz = full_df.loc[frame, f"{p_name}_vz"]
                        bvx = full_df.loc[frame, "ball_vx"]
                        bvy = full_df.loc[frame, "ball_vy"]
                        bvz = full_df.loc[frame, "ball_vz"]

                        p_speed = math.sqrt(pvx**2 + pvy**2 + pvz**2) if pd.notna(pvx) else 0
                        b_speed = math.sqrt(bvx**2 + bvy**2 + bvz**2) if pd.notna(bvx) else 0

                        hits_df.at[idx, 'Player_Speed'] = p_speed
                        hits_df.at[idx, 'Ball_Speed'] = b_speed
                        hits_df.at[idx, 'Player_VX'] = pvx
                        hits_df.at[idx, 'Player_VY'] = pvy
                        hits_df.at[idx, 'Boost_Amount'] = full_df.loc[frame, f"{p_name}_boost"]

                        # Геометрія (Відстані до воріт)
                        if p_color == 'royalblue':
                            hits_df.at[idx, 'Dist_to_Own_Goal'] = math.sqrt(px**2 + (py - (-5120))**2)
                            hits_df.at[idx, 'Dist_to_Opp_Goal'] = math.sqrt(px**2 + (py - 5120)**2)
                        else:
                            hits_df.at[idx, 'Dist_to_Own_Goal'] = math.sqrt(px**2 + (py - 5120)**2)
                            hits_df.at[idx, 'Dist_to_Opp_Goal'] = math.sqrt(px**2 + (py - (-5120))**2)

                        teammate_dists, opponent_dists, teammate_y_pos = [], [], [py]
                        opp_dists_to_ball = []
                        teammates_behind = 0

                        for other_p, other_c in player_colors.items():
                            if other_p == p_name or f"{other_p}_x" not in full_df.columns: continue
                            ox = full_df.loc[frame, f"{other_p}_x"]
                            oy = full_df.loc[frame, f"{other_p}_y"]
                            oz = full_df.loc[frame, f"{other_p}_z"]
                            if pd.isna(ox) or pd.isna(oy): continue
                        
                            dist = math.sqrt((ox - px)**2 + (oy - py)**2)
                            dist_to_ball = math.sqrt((ox - bx)**2 + (oy - by)**2 + (oz - bz)**2)

                            if other_c == p_color:
                                teammate_dists.append(dist)
                                teammate_y_pos.append(oy)
                                # Скільки тімейтів ближче до власних воріт, ніж м'яч
                                if (p_color == 'royalblue' and oy < by) or (p_color == 'darkorange' and oy > by):
                                    teammates_behind += 1
                            else:
                                opponent_dists.append(dist)
                                opp_dists_to_ball.append(dist_to_ball)
                                
                        hits_df.at[idx, 'Nearest_Teammate'] = min(teammate_dists) if teammate_dists else 9999
                        hits_df.at[idx, 'Nearest_Opponent'] = min(opponent_dists) if opponent_dists else 9999
                        hits_df.at[idx, 'Opp_Dist_To_Ball'] = min(opp_dists_to_ball) if opp_dists_to_ball else 9999
                        hits_df.at[idx, 'Teammates_Behind_Ball'] = teammates_behind
                        
                        is_last = 0
                        if len(teammate_y_pos) > 1:
                            if p_color == 'royalblue' and py == min(teammate_y_pos): is_last = 1
                            elif p_color == 'darkorange' and py == max(teammate_y_pos): is_last = 1
                        hits_df.at[idx, 'Is_Last_Man'] = is_last

            if ai_coach is not None and not hits_df.empty:
                try:
                    V4_FEATURES = [
                    # Координати
                    'Ball_X', 'Ball_Y', 'Ball_Z',
                    'Player_X', 'Player_Y', 'Player_Z',

                    # Швидкості
                    'Player_Speed', 'Ball_Speed',
                    'Player_VX', 'Player_VY',
                    'Ball_VX', 'Ball_VY', 'Ball_VZ',
                    'Avg_Teammate_Speed', 'Avg_Opponent_Speed',

                    # Геометрія
                    'Dist_to_Own_Goal', 'Dist_to_Opp_Goal',
                    'Dist_Ball_to_Own_Goal', 'Dist_Ball_to_Opp_Goal',
                    'Angle_Ball_to_Own_Goal',

                    # Команда
                    'Nearest_Teammate', 'Nearest_Opponent',
                    'Opp_Dist_To_Ball',
                    'Teammates_Ahead', 'Teammates_Behind_Ball',
                    'Is_Last_Man', 'Opp_Blocking',

                    # Ресурси
                    'Boost_Amount', 'Nearest_Boost_Pad',
                    'Is_Aerial', 'Is_High_Aerial',
                    'Going_Towards_Ball',

                    # Механіки
                    'Mech_On_Ground', 'Mech_On_Wall', 'Mech_On_Ceiling',
                    'Mech_Low_Aerial', 'Mech_Mid_Aerial', 'Mech_High_Aerial',
                    'Mech_Ball_Aerial', 'Mech_Ball_On_Wall', 'Mech_Ball_Ceiling',
                    'Mech_Ball_Above_Car',
                    'Mech_Is_Flick', 'Mech_Is_Aerial_Hit', 'Mech_Is_Ceiling_Shot',
                    'Mech_Is_Save', 'Mech_Is_Wall_Save',
                    'Mech_Is_Pinch', 'Mech_Is_Kuxir_Pinch',
                    'Mech_Ball_Backboard', 'Mech_Is_Air_Dribble',
                    'Mech_Is_50_50', 'Mech_Is_Supersonic', 'Mech_Is_Kickoff_Touch',
                    'Mech_High_Speed_Ground', 'Mech_Flip_Reset',
                    'Mech_Is_Psycho', 'Mech_Is_Fake',
                    'Mech_Player_Z', 'Mech_Ball_Z_Relative',
                    'Mech_Horiz_Speed', 'Mech_Vertical_Vel',
                ]

                    V3_FEATURES = [
                        'Ball_X', 'Ball_Y', 'Ball_Z',
                        'Player_X', 'Player_Y', 'Player_Z',
                        'Dist_to_Own_Goal', 'Dist_to_Opp_Goal',
                        'Nearest_Teammate', 'Nearest_Opponent', 'Opp_Dist_To_Ball',
                        'Teammates_Behind_Ball', 'Is_Last_Man',
                        'Boost_Amount', 'Player_Speed', 'Ball_Speed',
                        'Player_VY', 'Player_VX'
                    ]

                    # Додаємо відсутні колонки з нулями
                    all_needed = V4_FEATURES if AI_VERSION == 4 else V3_FEATURES
                    for col in all_needed:
                        if col not in hits_df.columns:
                            hits_df[col] = 0.0

                    # Також додаємо нові фічі V4 якщо їх немає
                    for col in ['Avg_Teammate_Speed', 'Avg_Opponent_Speed',
                                'Dist_Ball_to_Own_Goal', 'Dist_Ball_to_Opp_Goal',
                                'Angle_Ball_to_Own_Goal', 'Teammates_Ahead',
                                'Opp_Blocking', 'Nearest_Boost_Pad',
                                'Is_Aerial', 'Is_High_Aerial', 'Going_Towards_Ball',
                                'Ball_VX', 'Ball_VY', 'Ball_VZ']:
                        if col not in hits_df.columns:
                            hits_df[col] = 0.0

                    # Рахуємо нові фічі V4 для кожного рядка
                    BOOST_PAD_POSITIONS = [
                        (-3072, -4096), (3072, -4096),
                        (-3584, 0),     (3584, 0),
                        (-3072,  4096), (3072,  4096),
                        (0, -4096), (0, 4096),
                        (-1792, -2944), (1792, -2944),
                        (-1792,  2944), (1792,  2944),
                    ]

                    for idx, row in hits_df.iterrows():
                        frame = int(row['Frame'])
                        p_name = row['Player']
                        p_color = player_colors.get(p_name, 'royalblue')

                        if frame not in full_df.index:
                            continue

                        px = row.get('Player_X', 0) or 0
                        py = row.get('Player_Y', 0) or 0
                        bx = row.get('Ball_X', 0) or 0
                        by = row.get('Ball_Y', 0) or 0
                        bz = row.get('Ball_Z', 0) or 0

                        bvx = full_df.loc[frame, 'ball_vx'] if 'ball_vx' in full_df.columns else 0
                        bvy = full_df.loc[frame, 'ball_vy'] if 'ball_vy' in full_df.columns else 0
                        bvz = full_df.loc[frame, 'ball_vz'] if 'ball_vz' in full_df.columns else 0

                        hits_df.at[idx, 'Ball_VX'] = bvx
                        hits_df.at[idx, 'Ball_VY'] = bvy
                        hits_df.at[idx, 'Ball_VZ'] = bvz

                        own_goal_y = -5120 if p_color == 'royalblue' else 5120
                        opp_goal_y =  5120 if p_color == 'royalblue' else -5120

                        hits_df.at[idx, 'Dist_Ball_to_Own_Goal'] = math.sqrt(bx**2 + (by - own_goal_y)**2)
                        hits_df.at[idx, 'Dist_Ball_to_Opp_Goal'] = math.sqrt(bx**2 + (by - opp_goal_y)**2)
                        hits_df.at[idx, 'Angle_Ball_to_Own_Goal'] = abs(bx) / max(abs(own_goal_y - by), 1)

                        hits_df.at[idx, 'Is_Aerial']      = 1 if bz > 300 else 0
                        hits_df.at[idx, 'Is_High_Aerial'] = 1 if bz > 1000 else 0

                        pvx = full_df.loc[frame, f"{p_name}_vx"] if f"{p_name}_vx" in full_df.columns else 0
                        pvy = full_df.loc[frame, f"{p_name}_vy"] if f"{p_name}_vy" in full_df.columns else 0
                        dot = pvx * (bx - px) + pvy * (by - py)
                        hits_df.at[idx, 'Going_Towards_Ball'] = 1 if dot > 0 else 0

                        nearest_boost = min(
                            math.sqrt((px - bpx)**2 + (py - bpy)**2)
                            for bpx, bpy in BOOST_PAD_POSITIONS
                        )
                        hits_df.at[idx, 'Nearest_Boost_Pad'] = nearest_boost

                        tm_speeds, op_speeds = [], []
                        teammates_ahead = 0
                        opp_blocking = 0

                        for other_p, other_c in player_colors.items():
                            if other_p == p_name or f"{other_p}_x" not in full_df.columns:
                                continue
                            ox = full_df.loc[frame, f"{other_p}_x"]
                            oy = full_df.loc[frame, f"{other_p}_y"]
                            if pd.isna(ox) or pd.isna(oy):
                                continue

                            ovx = full_df.loc[frame, f"{other_p}_vx"] if f"{other_p}_vx" in full_df.columns else 0
                            ovy = full_df.loc[frame, f"{other_p}_vy"] if f"{other_p}_vy" in full_df.columns else 0
                            ovz = full_df.loc[frame, f"{other_p}_vz"] if f"{other_p}_vz" in full_df.columns else 0
                            o_speed = math.sqrt(ovx**2 + ovy**2 + ovz**2) if pd.notna(ovx) else 0

                            if other_c == p_color:
                                tm_speeds.append(o_speed)
                                if p_color == 'royalblue' and oy > by: teammates_ahead += 1
                                elif p_color == 'darkorange' and oy < by: teammates_ahead += 1
                            else:
                                op_speeds.append(o_speed)
                                if p_color == 'royalblue' and oy < py and abs(ox - px) < 800:
                                    opp_blocking = 1
                                elif p_color == 'darkorange' and oy > py and abs(ox - px) < 800:
                                    opp_blocking = 1

                        hits_df.at[idx, 'Avg_Teammate_Speed'] = np.mean(tm_speeds) if tm_speeds else 0
                        hits_df.at[idx, 'Avg_Opponent_Speed'] = np.mean(op_speeds) if op_speeds else 0
                        hits_df.at[idx, 'Teammates_Ahead']    = teammates_ahead
                        hits_df.at[idx, 'Opp_Blocking']       = opp_blocking

                    # Передбачення
                    active_features = V4_FEATURES if AI_VERSION == 4 else V3_FEATURES
                    X_all = hits_df[active_features].fillna(0)
                    
                    if AI_VERSION == 4:
                        probs_all = ai_coach.predict_proba(X_all)
                        hits_df['AI_Score']       = probs_all.max(axis=1) * 100
                        hits_df['AI_Action']      = probs_all.argmax(axis=1)
                        hits_df['AI_Action_Name'] = hits_df['AI_Action'].map({
                            0: 'Втрата', 1: 'Пас тімейту', 2: 'Збір бусту',
                            3: 'Удар по воротах', 4: 'Клір', 5: 'Бекборд пас',
                            6: 'Дриблінг', 7: 'Збереження позиції',
                            8: 'Гольова ситуація'
                        })
                        # Для сумісності з логікою помилок/успіхів
                        # Втрата = поганий мув, решта = добрий
                        hits_df['AI_Is_Good'] = hits_df['AI_Action'] != 0
                    else:
                        hits_df['AI_Score']   = ai_coach.predict_proba(X_all)[:, 1] * 100
                        hits_df['AI_Is_Good'] = hits_df['AI_Score'] > 50

                except Exception as e:
                    st.warning(f"⚠️ Помилка AI: {e}")

            player_total_xg = {}
            for p in match_stats:
                name = p['Name']
                allowed_shots = max(p.get('Shots', 0), p.get('Goals', 0))
                if name in player_all_touches_xg and allowed_shots > 0:
                    top_touches = sorted(player_all_touches_xg[name], reverse=True)
                    player_total_xg[name] = sum(top_touches[:allowed_shots])
                else:
                    player_total_xg[name] = 0.0

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
                else: st.info("Гравців не знайдено")
                    
            with col_o:
                st.markdown(f"### 🟠 Помаранчеві ({orange_names})")
                if not orange_team.empty:
                    styled_orange = orange_team.style.highlight_max(subset=['Rating 2.0'], color='#2e7d32').format(format_dict)
                    st.dataframe(styled_orange, width='stretch', hide_index=True)
                else: st.info("Гравців не знайдено")
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
                if len(members) < 1: continue 
                    
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
                            "Час на 0%": round((levels.count(0) / len(levels)) * 100, 1), 
                            "Колір": player_colors.get(name, 'cyan')
                        })
                
                if boost_stats:
                    fig_b, (ax_b1, ax_b2) = plt.subplots(2, 1, figsize=(6, 5.5)) 
                    fig_b.patch.set_alpha(0.0)
                    
                    df_boost = pd.DataFrame(boost_stats).sort_values("Середній буст", ascending=True)
                    ax_b1.patch.set_alpha(0.0)
                    bars1 = ax_b1.barh(df_boost["Гравець"], df_boost["Середній буст"], color=df_boost["Колір"])
                    ax_b1.bar_label(bars1, padding=3, color='white', fontsize=10, fmt='%.1f')
                    ax_b1.set_xlim(0, 105)
                    ax_b1.set_title("Середній рівень у баку (%)", color='white', pad=10, fontweight='bold')
                    ax_b1.spines['top'].set_visible(False); ax_b1.spines['right'].set_visible(False); ax_b1.spines['left'].set_visible(False)
                    ax_b1.spines['bottom'].set_color('gray')
                    ax_b1.tick_params(colors='white')
                    
                    df_starve = pd.DataFrame(boost_stats).sort_values("Час на 0%", ascending=True)
                    ax_b2.patch.set_alpha(0.0)
                    bars2 = ax_b2.barh(df_starve["Гравець"], df_starve["Час на 0%"], color='#ff4b4b')
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
                    fig_time, ax_time = plt.subplots(figsize=(6, 4))
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
            # 6. AI COACH: ТАКТИЧНА ДОШКА (FREEZE-FRAME)
            # ==========================================
            st.subheader("🗺️ AI Coach: Тактична дошка (Freeze-Frame)")
            st.caption("Оберіть момент для детального візуального та текстового розбору рішень ШІ.")

            analyze_options_dict = {}

            if 'AI_Score' in hits_df.columns:
                active_play_df = hits_df[
                    ~((hits_df['Ball_X'].abs() < 1500) & (hits_df['Ball_Y'].abs() < 1500))
                ].copy()

                if AI_VERSION == 4:
                    bad_plays  = active_play_df[active_play_df['AI_Action'] == 0].sort_values('AI_Score')
                    good_plays = active_play_df[
                        (active_play_df['AI_Action'].isin([1, 3, 5, 6, 8])) &  # додали 8
                        (active_play_df['AI_Score'] > 70)
                    ].sort_values('AI_Score', ascending=False)
                    
                    # Гольові ситуації окремо — показуємо навіть з низьким score
                    # (щоб бачити коли гравець МАВ бити але не вдарив)
                    missed_goals = active_play_df[
                        (active_play_df['AI_Action'] == 8) &
                        (active_play_df['AI_Score'] < 50)
                    ].sort_values('AI_Score')

                ACTION_ICONS = {
                    0: '🚨', 1: '🎯', 2: '⚡',
                    3: '🚀', 4: '🛡️', 5: '🏀', 6: '🔥', 7: '✅',
                }

                if not bad_plays.empty:
                    if not missed_goals.empty:
                        for idx, row in missed_goals.head(3).iterrows():
                            time_sec = int(row['Frame'] / 30)
                            time_str = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
                            label = f"🥅 Незабитий гол | {time_str} | {row['Player']}"
                            while label in analyze_options_dict: label += " "
                            analyze_options_dict[label] = {
                                'frame': int(row['Frame']),
                                'player': row['Player'],
                                'score': row['AI_Score'],
                                'action': 'Гольова ситуація',
                            }

                if not good_plays.empty:
                    for idx, row in good_plays.head(5).iterrows():
                        time_sec = int(row['Frame'] / 30)
                        time_str = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
                        action_id = int(row.get('AI_Action', 7))
                        icon = ACTION_ICONS.get(action_id, '✅')
                        label = f"{icon} {row.get('AI_Action_Name','Успіх')} | {time_str} | {row['Player']}"
                        while label in analyze_options_dict: label += " "
                        analyze_options_dict[label] = {
                            'frame': int(row['Frame']),
                            'player': row['Player'],
                            'score': row['AI_Score'],
                            'action': row.get('AI_Action_Name', 'Успіх'),
                        }


            if analyze_options_dict:
                selected_label = st.selectbox("Оберіть момент для розбору:", list(analyze_options_dict.keys()))
                
                target_frame = analyze_options_dict[selected_label]['frame']
                target_player = analyze_options_dict[selected_label]['player']
                score = analyze_options_dict[selected_label]['score']
                
                col_text, col_plot = st.columns([1.5, 1.5])
                
                with col_text:
                    moment_data = hits_df[(hits_df['Frame'] == target_frame) & (hits_df['Player'] == target_player)]
                    if not moment_data.empty:
                        row = moment_data.iloc[0]
                        
                        action_name = analyze_options_dict[selected_label].get('action', 'Невідомо')

                        if AI_VERSION == 4:
                            ACTION_ICONS = {
                                'Втрата': '🚨', 'Пас тімейту': '🎯', 'Збір бусту': '⚡',
                                'Удар по воротах': '🚀', 'Клір': '🛡️', 'Бекборд пас': '🏀',
                                'Дриблінг': '🔥', 'Збереження позиції': '✅',
                            }
                            icon = ACTION_ICONS.get(action_name, '🤖')
                            st.markdown(f"### {icon} ШІ визначив: **{action_name}**")
                            st.metric("Впевненість моделі", f"{score:.1f}%")
                        else:
                            st.markdown(f"### 🤖 Розбір від ШІ")
                            st.metric("Шанс успіху", f"{score:.1f}%")
                        
                        st.markdown("#### Що зафіксували сенсори (V3):")
                        st.write(f"🔋 **Буст:** {int(row['Boost_Amount'])}%")
                        st.write(f"🏎️ **Швидкість:** {int(row['Player_Speed'])} | ⚽ **М'яч:** {int(row['Ball_Speed'])}")
                        
                        teammate_txt = f"{int(row['Teammates_Behind_Ball'])} гравців" if row['Teammates_Behind_Ball'] > 0 else "НІКОГО (Пусті ворота!)"
                        st.write(f"🛡️ **Страховка позаду:** {teammate_txt}")
                        opp_ball_dist = row['Opp_Dist_To_Ball'] / 100
                        st.write(f"⚔️ **Суперник:** за {row['Nearest_Opponent'] / 100:.1f} м (Відстань ворога до м'яча: {opp_ball_dist:.1f} м)")
                        
                        # --- ДОДАНИЙ БЛОК: Механіки детектор ---
                        detected_mechs = []
                        mech_display = {
                            'Mech_Is_Flick':          '🤸 Флік',
                            'Mech_Is_Ceiling_Shot':   '🏠 Ceiling Shot',
                            'Mech_Is_Air_Dribble':    '🌀 Air Dribble',
                            'Mech_Is_Pinch':          '💥 Пінч',
                            'Mech_Is_Kuxir_Pinch':    '⚡ Kuxir Pinch',
                            'Mech_Ball_Backboard':    '🏀 Backboard',
                            'Mech_Is_50_50':          '⚔️ 50/50',
                            'Mech_Is_Save':           '🧤 Сейв',
                            'Mech_Is_Wall_Save':      '🧱 Wall Save',
                            'Mech_Is_Supersonic':     '🚀 Supersonic',
                            'Mech_Is_Psycho':         '🧠 Psycho Shot',
                            'Mech_Flip_Reset':        '🔄 Flip Reset',
                            'Mech_Is_Fake':           '🎭 Фейк',
                            'Mech_On_Ceiling':        '🏠 На стелі',
                            'Mech_On_Wall':           '🧱 На стіні',
                            'Mech_High_Aerial':       '✈️ High Aerial',
                        }
                        for col, label in mech_display.items():
                            if col in row and row[col] == 1:
                                detected_mechs.append(label)

                        if detected_mechs:
                            st.markdown("#### 🎮 Виявлені механіки:")
                            st.info("  |  ".join(detected_mechs))
                        else:
                            st.caption("🎮 Стандартний дотик, без спеціальних механік")
                        # ----------------------------------------
                        
                        st.markdown("#### Вердикт:")
                        # НОВИЙ ВЕРДИКТ V4
                        ACTION_LABELS = {
                            0: 'Втрата', 1: 'Пас тімейту', 2: 'Збір бусту',
                            3: 'Удар по воротах', 4: 'Клір', 5: 'Бекборд пас',
                            6: 'Дриблінг', 7: 'Збереження позиції',
                            8: 'Гольова ситуація',  # НОВЕ для V4
                        }
                        ACTION_ICONS = {
                            0: '❌', 1: '🎯', 2: '⚡', 3: '🚀', 
                            4: '🛡️', 5: '🏀', 6: '🔥', 7: '✅',
                             8: '🥅',  # НОВЕ
                        }

                        if AI_VERSION == 4 and ai_coach is not None:
                            try:
                                V4_FEATURES = [
                                # Координати
                                'Ball_X', 'Ball_Y', 'Ball_Z',
                                'Player_X', 'Player_Y', 'Player_Z',

                                # Швидкості
                                'Player_Speed', 'Ball_Speed',
                                'Player_VX', 'Player_VY',
                                'Ball_VX', 'Ball_VY', 'Ball_VZ',
                                'Avg_Teammate_Speed', 'Avg_Opponent_Speed',

                                # Геометрія
                                'Dist_to_Own_Goal', 'Dist_to_Opp_Goal',
                                'Dist_Ball_to_Own_Goal', 'Dist_Ball_to_Opp_Goal',
                                'Angle_Ball_to_Own_Goal',

                                # Команда
                                'Nearest_Teammate', 'Nearest_Opponent',
                                'Opp_Dist_To_Ball',
                                'Teammates_Ahead', 'Teammates_Behind_Ball',
                                'Is_Last_Man', 'Opp_Blocking',

                                # Ресурси
                                'Boost_Amount', 'Nearest_Boost_Pad',
                                'Is_Aerial', 'Is_High_Aerial',
                                'Going_Towards_Ball',

                                # Механіки
                                'Mech_On_Ground', 'Mech_On_Wall', 'Mech_On_Ceiling',
                                'Mech_Low_Aerial', 'Mech_Mid_Aerial', 'Mech_High_Aerial',
                                'Mech_Ball_Aerial', 'Mech_Ball_On_Wall', 'Mech_Ball_Ceiling',
                                'Mech_Ball_Above_Car',
                                'Mech_Is_Flick', 'Mech_Is_Aerial_Hit', 'Mech_Is_Ceiling_Shot',
                                'Mech_Is_Save', 'Mech_Is_Wall_Save',
                                'Mech_Is_Pinch', 'Mech_Is_Kuxir_Pinch',
                                'Mech_Ball_Backboard', 'Mech_Is_Air_Dribble',
                                'Mech_Is_50_50', 'Mech_Is_Supersonic', 'Mech_Is_Kickoff_Touch',
                                'Mech_High_Speed_Ground', 'Mech_Flip_Reset',
                                'Mech_Is_Psycho', 'Mech_Is_Fake',
                                'Mech_Player_Z', 'Mech_Ball_Z_Relative',
                                'Mech_Horiz_Speed', 'Mech_Vertical_Vel',
                            ]
                                # Беремо фічі з поточного рядка row
                                x_single = pd.DataFrame([row])[V4_FEATURES].fillna(0)
                                probs = ai_coach.predict_proba(x_single)[0]
                                
                                # Топ-3 варіанти
                                st.markdown("#### 🎲 Що це могло бути:")
                                top3 = np.argsort(probs)[::-1][:3]
                                for i in top3:
                                    bar_pct = int(probs[i] * 100)
                                    if bar_pct < 1:
                                        continue
                                    st.write(f"{ACTION_ICONS.get(i,'▪️')} **{ACTION_LABELS.get(i,'?')}**: {bar_pct}%")
                                    st.progress(bar_pct)
                            except Exception as e:
                                st.caption(f"Деталі недоступні: {e}")

                        st.markdown("#### ⏩ Що сталось далі:")
                        AFTER_FRAMES = 90
                        after_frame = min(target_frame + AFTER_FRAMES, len(full_df) - 1)

                        if "ball_y" in full_df.columns and after_frame in full_df.index:
                            ball_y_now   = full_df.loc[target_frame, "ball_y"]
                            ball_y_after = full_df.loc[after_frame,  "ball_y"]

                            # Перевіряємо чи був гол — м'яч перейшов лінію воріт (|y| > 5000)
                            # або різко повернувся в центр (телепорт після голу)
                            goal_scored = False
                            goal_scorer_team = None

                            # Сканує 90 фреймів після дотику
                            for check_frame in range(target_frame, min(target_frame + 90, len(full_df))):
                                if check_frame not in full_df.index:
                                    continue
                                check_by = full_df.loc[check_frame, "ball_y"]
                                if pd.isna(check_by):
                                    continue

                                # М'яч за лінією воріт
                                if check_by > 5050:
                                    goal_scored = True
                                    goal_scorer_team = 'royalblue'  # Blue забив (м'яч у воротах Orange)
                                    break
                                if check_by < -5050:
                                    goal_scored = True
                                    goal_scorer_team = 'darkorange'  # Orange забив
                                    break

                                # Телепорт у центр = гол (м'яч стрибнув з атакуючої зони до |y| < 500)
                                if check_frame > target_frame + 5:
                                    prev_by = full_df.loc[check_frame - 1, "ball_y"] if (check_frame - 1) in full_df.index else check_by
                                    if pd.notna(prev_by):
                                        was_attacking_blue   = prev_by > 4000
                                        was_attacking_orange = prev_by < -4000
                                        teleported_to_center = abs(check_by) < 800

                                        if was_attacking_blue and teleported_to_center:
                                            goal_scored = True
                                            goal_scorer_team = 'royalblue'
                                            break
                                        if was_attacking_orange and teleported_to_center:
                                            goal_scored = True
                                            goal_scorer_team = 'darkorange'
                                            break

                            p_color = player_colors.get(target_player, 'royalblue')

                            if goal_scored:
                                if goal_scorer_team == p_color:
                                    st.success("🎉 **ГОЛ!** Удар досяг мети — м'яч у сітці!")
                                else:
                                    st.error("💀 **ГОЛ ПРОПУЩЕНО!** Суперник після цього моменту забив.")

                            elif pd.notna(ball_y_now) and pd.notna(ball_y_after):
                                delta_y = ball_y_after - ball_y_now

                                if action_name == 'Гольова ситуація' and not goal_scored:
                                    st.warning("😤 **Момент був — але гол не вийшов.** М'яч не перетнув лінію воріт.")

                                elif score < 50:
                                    if abs(delta_y) < 500:
                                        st.warning("😐 М'яч залишився в тій же зоні — суперник не скористався, але могло бути гірше.")
                                    elif p_color == 'royalblue' and delta_y < -800:
                                        st.error("💀 М'яч пішов у твої ворота — суперник скористався помилкою!")
                                    elif p_color == 'darkorange' and delta_y > 800:
                                        st.error("💀 М'яч пішов у твої ворота — суперник скористався помилкою!")
                                    else:
                                        st.info("🛡️ Команда відбила загрозу попри помилку.")
                                else:
                                    if p_color == 'royalblue' and delta_y > 800:
                                        st.success("🚀 М'яч пішов у атаку — хороший мув створив тиск!")
                                    elif p_color == 'darkorange' and delta_y < -800:
                                        st.success("🚀 М'яч пішов у атаку — хороший мув створив тиск!")
                                    else:
                                        st.info("🔄 М'яч повернули, але суперник встиг перегрупуватись.")

                            next_hits = hits_df[
                                (hits_df['Frame'] > target_frame) &
                                (hits_df['Frame'] < target_frame + 300)
                            ]
                            if not next_hits.empty:
                                frames_to_next = next_hits.iloc[0]['Frame'] - target_frame
                                secs = frames_to_next // 30
                                st.caption(f"⚡ Наступний дотик: через **{secs} сек** — {next_hits.iloc[0]['Player']}")


                with col_plot:
                    gif_cache_key = f"gif_{target_frame}_{target_player}"
                    
                    # Генеруємо тільки якщо ще не генерували
                    if gif_cache_key not in st.session_state:
                        with st.spinner("🎬 Генерація анімації..."):
                            gif_path = generate_moment_gif(
                                full_df, player_colors, 
                                target_frame, list(player_colors.keys())
                            )
                            # Читаємо в байти щоб зберегти в session_state
                            with open(gif_path, 'rb') as f:
                                st.session_state[gif_cache_key] = f.read()
                            import os
                            os.unlink(gif_path)  # видаляємо тимчасовий файл
                    
                    st.image(
                        st.session_state[gif_cache_key], 
                        caption=f"{'🚨 Помилка' if score < 50 else '✅ Успіх'} | {target_player}",
                        use_container_width=True
                    )
                    
                    # Кнопка для перегенерації
                    if st.button("🔄 Оновити анімацію", key=f"regen_{target_frame}"):
                        del st.session_state[gif_cache_key]
                        st.rerun()
            else:
                st.info("ШІ не знайшов цікавих моментів для розбору в цьому матчі.")
else:
    st.info("Будь ласка, завантажте файл .replay у бічній панелі для генерації повного дашборду.")