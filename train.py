"""Builds the artifacts the API serves from.

Reads the raw dataset, scales the feature columns, and saves two files:
- features_scaled.joblib: scaled feature matrix (one row per song)
- song_metadata.joblib: track_name / artist_name / genre / release_date,
  in the same row order as the feature matrix

Run this once (and again any time the dataset or FEATURE_COLUMNS changes)
before starting the API.
"""

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/tcc_ceds_music.csv")
FEATURES_PATH = Path("artifacts/features_scaled.joblib")
METADATA_PATH = Path("artifacts/song_metadata.joblib")

FEATURE_COLUMNS = [
    "dating", "violence", "world/life", "night/time",
    "shake the audience", "family/gospel", "romantic",
    "communication", "obscene", "music", "movement/places",
    "light/visual perceptions", "family/spiritual", "like/girls",
    "sadness", "feelings",
    "danceability", "loudness", "acousticness",
    "instrumentalness", "valence", "energy",
]

METADATA_COLUMNS = ["track_name", "artist_name", "genre", "release_date"]


def build_artifacts() -> None:
    data = pd.read_csv(DATA_PATH)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data[FEATURE_COLUMNS])

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(features_scaled, FEATURES_PATH)
    dump(data[METADATA_COLUMNS], METADATA_PATH)

    print(f"Saved {len(data)} songs to {FEATURES_PATH} and {METADATA_PATH}")


if __name__ == "__main__":
    build_artifacts()
