# Music Recommender API

A FastAPI service that recommends songs similar to a given song, using cosine similarity over lyrical and audio features.

**Live:** https://music-recommender-api.fastapicloud.dev/docs

This is the sequel to [content-based-music-recommender](https://github.com/Ali-reza-rn/content-based-music-recommender). That project is the analysis: it works out *how* to compare songs. This one is the engineering: it turns the notebook into a service with validated requests, typed responses, and tests.

> Just want to run it? Jump to [Getting started](#getting-started).

## Why StandardScaler?

The model describes each of the 28,372 songs in the dataset with 22 numbers (16 lyrical-theme scores and 6 audio features) and compares songs with cosine similarity. The first version had a problem that was easy to miss, because it still returned recommendations.

**Without scaling, every song looked similar to every other song.** The top-10 recommendations for "Sweater Weather" all scored between 0.979 and 0.985. The model could not tell a great match from a mediocre one.

I tried three fixes, in this order:

| Attempt | What happened |
|---|---|
| No scaling | Scores squeezed into 0.979-0.985 for "Sweater Weather". The model can't separate songs. |
| `MinMaxScaler` | Barely improved anything. |
| PCA (15 components) | Spread most scores out, but gave a few pairs artificially high scores. For "Love You Like a Love Song" and "Basketball", the 15 kept components scored 0.972 while the 7 discarded ones scored -0.174. |
| **`StandardScaler`** | Scores spread out into a meaningful range, with no dimensionality reduction. |

**Why it works.** Cosine similarity compares the *direction* of two vectors. All 22 features are non-negative and roughly in the 0-1 range, so every song's vector points into the same corner of the space. Any two of them have a small angle between them, and therefore a high cosine similarity, no matter how different the songs are. The 16 lyrical features are also sparse (most values sit near zero), which makes this worse.

`StandardScaler` subtracts each feature's mean, so values land above and below zero. Vectors can now point in genuinely different directions, and cosine similarity starts measuring real differences. It also puts every feature on a comparable scale: the features are only approximately normalized, and their real ranges differ (the maximum of `shake the audience` is 0.497, while `obscene` reaches 0.992).

**Why not PCA?** My first hypothesis was that 22 dimensions was too many. It wasn't. The first principal component was dominated by `acousticness`, `energy` and `valence`, while the 16 lyrical features contributed almost nothing. For one high-scoring pair (Love You Like a Love Song and Basketball), similarity on the 15 kept components was 0.972, but on the 7 discarded components it was -0.174. PCA had thrown away real signal just because it had low variance.

**The result.** After `StandardScaler`, the top-10 scores for "Sweater Weather" range from 0.743 to 0.800, and for "Love You Like a Love Song" from 0.805 to 0.888. The two ranges don't overlap at all, so the model now separates these two thematically different songs. The full experiments, plots and code are in the [original repository](https://github.com/Ali-reza-rn/content-based-music-recommender).

## How it works

```
data/tcc_ceds_music.csv
        |
        v   train.py  (run once, or to rebuild)
artifacts/features_scaled.joblib   <- StandardScaler output, one row per song
artifacts/song_metadata.joblib     <- track_name, artist_name, genre, release_date
        |
        v   app/main.py  (loaded once at startup via `lifespan`)
FastAPI: /search, /recommend, /health
```

- `train.py` reads the dataset, standardizes the 22 feature columns and saves two files. The feature matrix and the metadata share the same row order, which is how a row number found in one is translated back into a song in the other.
- `app/main.py` loads both files once at startup and keeps them in `app.state`. Nothing is read from disk or retrained per request.
- The scaler itself is not saved. The API only recommends songs that are already in the dataset, so it never needs to transform new data.
- Similarity is computed on demand, for one song against all the others. A full precomputed similarity matrix for 28,372 songs would need roughly 6 GB of memory, while a single row costs almost nothing.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check that the API is running |
| `GET` | `/search` | Find songs by title (and optionally artist) prefix |
| `POST` | `/recommend` | Get songs similar to a given song |

Interactive docs, with the full request and response schemas, are at `/docs`.

The examples below use the live deployment. To hit a local instance instead, replace `https://music-recommender-api.fastapicloud.dev` with `http://127.0.0.1:8000`.

### `GET /search`

Case-insensitive prefix matching. Use it to find the exact title and artist before calling `/recommend`.

```bash
curl -G "https://music-recommender-api.fastapicloud.dev/search" \
  --data-urlencode "song_title=shape of my heart"
```

```json
[
  {"track_name": "shape of my heart", "artist_name": "backstreet boys", "genre": "pop"},
  {"track_name": "shape of my heart", "artist_name": "sting", "genre": "rock"}
]
```

Returns `404` if nothing matches and `422` if `song_title` is empty.

### `POST /recommend`

```bash
curl -X POST "https://music-recommender-api.fastapicloud.dev/recommend" \
  -H "Content-Type: application/json" \
  -d '{"song_title": "sweater weather", "top_n": 5}'
```

```json
{
  "note": null,
  "song_title": "sweater weather",
  "artist_name": "the neighbourhood",
  "recommendations": [
    {"track_name": "tabloid junkie", "artist_name": "michael jackson", "genre": "pop", "similarity": 0.8},
    {"track_name": "keep on", "artist_name": "the green", "genre": "reggae", "similarity": 0.78},
    {"track_name": "breathe", "artist_name": "blu cantrell", "genre": "pop", "similarity": 0.774},
    {"track_name": "outrage! is now", "artist_name": "death from above 1979", "genre": "blues", "similarity": 0.77},
    {"track_name": "skin & bones", "artist_name": "eli young band", "genre": "country", "similarity": 0.765}
  ]
}
```

Songs are matched on their lyrical-theme and audio features only, not on genre. That is why the results above cross genre boundaries (see [Limitations](#limitations)).

| Field | Rules |
|---|---|
| `song_title` | Required, at least 1 character. Exact match, case-insensitive. |
| `artist_name` | Optional. Use it when several songs share a title. |
| `top_n` | Optional, default 10, between 1 and 25. |

- Several songs can share a title (for example "Shape of My Heart" by two different artists). The API then uses the first match and explains it in `note`, e.g. `Multiple matches found for 'shape of my heart': ['backstreet boys', 'sting']. Using 'backstreet boys'. Pass artist_name to disambiguate.`
- Invalid input returns `422`. A song that isn't in the dataset returns `404`.

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.12 or newer.

```bash
git clone https://github.com/Ali-reza-rn/music-recommender-api.git
cd music-recommender-api
uv sync
uv run fastapi dev
```

Then open <http://127.0.0.1:8000/docs>. The prebuilt artifacts are committed, so the API runs without downloading the dataset.

### Rebuilding the artifacts

You only need this if you change the features or the dataset.

1. Download `tcc_ceds_music.csv` from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019) and save it as `data/tcc_ceds_music.csv` (create the `data/` folder if needed).
2. Run:
   ```bash
   uv run python train.py
   ```
3. Commit the updated files in `artifacts/`.

The artifacts are `joblib` files, which are pickles, so they depend on the library versions that built them. After upgrading `pandas`, `scikit-learn` or `numpy`, rebuild them. If they are missing, the app refuses to start and tells you to run `train.py`.

## Project structure

```
.
├── app/
│   ├── __init__.py
│   └── main.py          # API: endpoints, request/response models
├── artifacts/           # built by train.py, committed so the API runs without the dataset
├── data/                # dataset goes here (the CSV is not committed)
├── tests/
│   ├── conftest.py      # shared `client` fixture
│   ├── test_recommend.py
│   └── test_search.py
├── train.py             # builds the artifacts
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## Tests

```bash
uv run pytest
```

The suite uses `pytest` and FastAPI's `TestClient`, with `lifespan` running so the real artifacts are loaded. It covers:

- valid requests, with the right song resolved and the right number of results
- validation errors (`422`): `top_n` of 0, negative or above the limit, empty or missing `song_title`
- unknown songs and unknown artists (`404`)
- case-insensitive matching on both endpoints
- ambiguous titles returning a `note`
- recommendations sorted by descending similarity, and the searched song not appearing in its own recommendations (checked on a sample query)
- the response shape and field types

## Design decisions

- **Artifacts built once, loaded at startup.** Training and serving are separate steps, so a request never triggers recomputation.
- **Typed requests and responses.** Pydantic models validate input (`song_title`, `top_n` bounds) and describe output, so `/docs` shows the real response schema and a malformed response fails loudly instead of being sent.
- **`GET` for search, `POST` for recommend.** Search is a read-only lookup, so it uses query parameters. Recommend takes a structured body.
- **Search is prefix matching, recommend is exact.** Search helps you find a title. Recommend needs one specific song.

## Limitations

- `genre` is not part of the feature vector, so recommendations are based purely on lyrical and audio similarity and can cross genre boundaries.
- Only songs already in the dataset can be used as input. Supporting new songs would mean saving the fitted scaler and computing features for them.
- When a title is ambiguous, the first match is used rather than asking the caller to choose.

## Future work

- Weight lyrical and audio features separately instead of equally.

## Dataset

[Music Dataset: 1950 to 2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019) from Kaggle: 28,372 songs with metadata, 16 lyrical-theme scores and 6 audio features.

## License

MIT. See [LICENSE](LICENSE). The dataset is not part of this repository and is covered by its own terms on its Kaggle page.
