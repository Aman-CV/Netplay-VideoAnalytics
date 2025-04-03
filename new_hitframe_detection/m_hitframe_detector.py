import pandas as pd
from scipy.ndimage import binary_closing
import numpy as np



def get_hitframes(dff):
    df = pd.DataFrame(dff)
    df = df.bfill(axis=0)
    print(df.columns)
    df['delta_y'] = df[3].diff().fillna(0)
    df['delta_y'] = df["delta_y"].abs()
    window_size = 10
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i + window_size]  # Get the current 10-element window
        if len(window) == window_size and window.iloc[0]["delta_y"] < 10 and window.iloc[-1]["delta_y"] < 10:
            df.iloc[i:i + window_size, df.columns.get_loc("delta_y")] = 0  # Modify only column "A"
    df["rolling_avg"] = df["delta_y"].rolling(window=10).mean().fillna(0)
    df["rolling_avg"] = df["rolling_avg"].rolling(window=10).mean().fillna(0)
    df["rolling_avg"] = df["rolling_avg"].rolling(window=10).mean().fillna(0)
    df["rolling_avg"].bfill()

    df["is_zero"] = (df["rolling_avg"] > 0.4).astype(int)
    df["smooth"] = pd.Series(binary_closing(df["is_zero"], structure=np.ones(50)).astype(int))
    return df["smooth"].to_numpy()

def clip_hitframes(video_frames, hitframes):
    hit_video_frames = []
    for frame, is_hit in zip(video_frames, hitframes):
        if is_hit:
            hit_video_frames.append(frame)
    return hit_video_frames

