"""FastAPI server for the content-based music recommender.

Serves recommendations from precomputed artifacts (see train.py):
- features_scaled.joblib: scaled feature matrix (one row per song)
- song_metadata.joblib: track_name / artist_name / genre / release_date,
  in the same row order as the feature matrix
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from joblib import load
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

FEATURES_PATH = Path("artifacts/features_scaled.joblib")
METADATA_PATH = Path("artifacts/song_metadata.joblib")

METADATA_COLUMNS = ["track_name", "artist_name", "genre"]


# Request models

class RecommendationRequest(BaseModel):
    song_title: str = Field(min_length=1)
    artist_name: str | None = Field(default=None, min_length=1)
    top_n: int = Field(default=10, gt=0, le=25)


# Response models

class SongInfo(BaseModel):
    """Shared shape for a song's basic metadata."""
    track_name: str
    artist_name: str
    genre: str


class RecommendationResult(SongInfo):
    similarity: float


class RecommendationResponse(BaseModel):
    note: str | None
    song_title: str
    artist_name: str
    recommendations: list[RecommendationResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the precomputed artifacts into app.state before serving requests.

    Runs once at startup. Raises RuntimeError if the artifact files are
    missing, so a misconfigured deployment fails immediately instead of
    on the first request.
    """
    if not FEATURES_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Artifact files not found. Run train.py first to generate "
            f"{FEATURES_PATH} and {METADATA_PATH}."
        )
    app.state.features = load(FEATURES_PATH)
    app.state.metadata = load(METADATA_PATH)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Content-Based Music Recommender",
    version="1.0.0",
    description="Serves song recommendations" \
    " from a precomputed similarity model.",
)


@app.get("/health")
def health():
    """Check that the API is running."""
    return {"status": "ok"}


@app.get("/search", response_model=list[SongInfo])
def search_songs(
    song_title: str = Query(min_length=1),
    artist_name: str | None = None,
):
    """Find songs whose title (and optionally artist)
    starts with the given text.

    Matching is case-insensitive prefix matching, so `sweater` finds
    "sweater weather". Use this to look up the exact title and artist
    before calling `/recommend`.

    Returns 404 if no song matches.
    """

    metadata = app.state.metadata
    song_title = song_title.lower()

    if artist_name:
        mask = (
            metadata["artist_name"].str.startswith(artist_name.lower())) & (
            metadata["track_name"].str.startswith(song_title)
        )
    else:
        mask = metadata["track_name"].str.startswith(song_title)

    matches = metadata.loc[mask, METADATA_COLUMNS].to_dict(orient="records")
    if matches:
        return matches
    raise HTTPException(status_code=404, detail="no song found")


def locate_song(
    metadata,
    song_title: str,
    artist_name: str | None,
) -> tuple[int, str | None]:
    """Find a song's row index in metadata by title (and optionally artist).

    Returns (row_index, note). note is None unless multiple songs matched
    the given title, in which case it explains which one was picked.
    Raises HTTPException(404) if no song matched.
    """
    song_title = song_title.lower()

    if artist_name:
        mask = (metadata["track_name"] == song_title) & (
            metadata["artist_name"] == artist_name.lower()
        )
    else:
        mask = metadata["track_name"] == song_title

    matches = metadata[mask].index

    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No song found matching '{song_title}'.",
        )

    note = None
    if len(matches) > 1:
        candidates = metadata.loc[matches, METADATA_COLUMNS]
        note = (
            f"Multiple matches found for '{song_title}': "
            f"{candidates['artist_name'].tolist()}. "
            f"Using '{metadata.loc[matches[0], 'artist_name']}'. "
            "Pass artist_name to disambiguate."
        )

    return matches[0], note


def rank_similar_songs(
    features,
    song_idx: int,
    top_n: int
    ) -> list[tuple[int, float]]:

    """Rank all songs by cosine similarity to the song at song_idx.

    Returns the top_n (row_index, similarity_score) pairs, highest score
    first, excluding the song itself.
    """
    scores = cosine_similarity(features[song_idx : song_idx + 1], features)[0]
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    # skip the song itself
    return [pair for pair in ranked if pair[0] != song_idx][:top_n]


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """Recommend songs similar to the given one, using cosine similarity.

    Similarity is computed over 22 standardized features (16 lyrical themes
    and 6 audio features). Matching on the title is exact (case-insensitive).

    If several songs share the title, the first match is used and `note`
    lists the candidates. Pass `artist_name` to pick a specific one.
    `top_n` is capped at 25.

    Returns 404 if the song is not found.
    """
    metadata = app.state.metadata
    features = app.state.features

    song_idx, note = locate_song(
        metadata, request.song_title, request.artist_name
    )
    ranked = rank_similar_songs(features, song_idx, request.top_n)

    result_indices = [idx for idx, _ in ranked]
    recommendations = metadata.iloc[result_indices][METADATA_COLUMNS].copy()
    recommendations["similarity"] = [round(score, 3) for _, score in ranked]

    return {
        "note": note,
        "song_title": request.song_title,
        "artist_name": metadata.loc[song_idx, "artist_name"],
        "recommendations": recommendations.to_dict(orient="records")
    }
