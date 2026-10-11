"""Tests for the /health and /recommend endpoints.

Covers valid requests, validation errors (422), unknown songs (404),
case-insensitive matching, ambiguous titles, similarity ordering, the
searched song never appearing in its own results, and response shape.
The `client` fixture comes from conftest.py.
"""

import pytest

TOP_N = 10
MAX_TOP_N = 25  # must match the `le=` limit on top_n in RecommendationRequest


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Happy path: valid requests return the expected song and shape

@pytest.mark.parametrize("request_body, expected_artist", [
    (
        {
            "song_title": "shape of my heart",
            "artist_name": "sting",
            "top_n": TOP_N
        },
        "sting",
    ),
    (
        {"song_title": "sweater weather", "artist_name": None, "top_n": TOP_N},
        "the neighbourhood",
    ),
])
def test_recommendation_returns_expected_song(
    client, request_body, expected_artist
    ):
    response = client.post("/recommend", json=request_body)
    assert response.status_code == 200

    body = response.json()
    assert len(body["recommendations"]) == request_body["top_n"]
    assert body["note"] is None
    if expected_artist is not None:
        assert body["artist_name"] == expected_artist


# Invalid input: requests that fail Pydantic validation

@pytest.mark.parametrize("request_body", [
    {"song_title": "sweater weather", "artist_name": None, "top_n": 0},
    {"song_title": "sweater weather", "artist_name": None, "top_n": -1},
    {
        "song_title": "sweater weather",
        "artist_name": None,
        "top_n": MAX_TOP_N + 1
    },
    {"song_title": "", "artist_name": None, "top_n": TOP_N},
    {"artist_name": None, "top_n": TOP_N},  # song_title missing entirely
], ids=[
    "top_n_zero",
    "top_n_negative",
    "top_n_above_max",
    "empty_song_title",
    "missing_song_title",
])
def test_recommendation_rejects_invalid_input(client, request_body):
    response = client.post("/recommend", json=request_body)
    assert response.status_code == 422


# Not found: valid input shape, but no matching song in the dataset

@pytest.mark.parametrize("request_body", [
    {
        "song_title": "a song that definitely does not exist",
        "artist_name": None,
        "top_n": TOP_N
    },
    {
        "song_title": "sweater weather",
        "artist_name": "an artist that does not exist",
        "top_n": TOP_N
    },
], ids=[
    "unknown_song_title",
    "unknown_artist_for_known_title",
])
def test_recommendation_not_found(client, request_body):
    response = client.post("/recommend", json=request_body)
    assert response.status_code == 404


# Case Insensitive
@pytest.mark.parametrize("request_body", [
    {"song_title": "SwEaTeR WeAtHeR", "artist_name": None, "top_n": TOP_N},
    {
        "song_title": "SwEaTeR WeAtHeR",
        "artist_name": "tHe neiGhbouRhOOd",
        "top_n": TOP_N
    }
])
def test_recommendation_is_case_insensitive(client, request_body):
    response = client.post("/recommend", json=request_body)
    assert response.status_code == 200


# Ambiguous match: multiple songs share a title

def test_recommendation_ambiguous_match_returns_note(client):
    request_body = {
        "song_title": "shape of my heart",
        "artist_name": None,
        "top_n": TOP_N,
    }
    response = client.post("/recommend", json=request_body)
    assert response.status_code == 200
    assert response.json()["note"] is not None


# Similarity is sorted:
def test_recommendation_similarity_is_sorted(client):
    request_body = {
        "song_title": "sweater weather",
        "artist_name": None,
        "top_n": TOP_N,
    }
    response = client.post("/recommend", json=request_body)
    body = response.json()
    similarity = [item["similarity"] for item in body["recommendations"]]
    assert similarity == sorted(similarity, reverse=True)


# The searched song must never appear in its own recommendations

@pytest.mark.parametrize("request_body", [
    {
        "song_title": "sweater weather",
        "artist_name": None,
        "top_n": MAX_TOP_N
    },
    {
        "song_title": "shape of my heart",
        "artist_name": "sting",
        "top_n": MAX_TOP_N
    },
    {
        "song_title": "love you like a love song",
        "artist_name": None,
        "top_n": MAX_TOP_N
    },
])
def test_recommendation_excludes_searched_song(client, request_body):
    response = client.post("/recommend", json=request_body)
    assert response.status_code == 200

    body = response.json()
    # Compare title AND artist, using the artist the API actually resolved.
    # Another artist's song with the same title is allowed to appear.
    searched_song = (request_body["song_title"].lower(), body["artist_name"])
    returned_songs = [
        (item["track_name"], item["artist_name"])
        for item in body["recommendations"]
    ]
    assert searched_song not in returned_songs


# Response shape: structure and types are correct

def test_recommendation_shape(client):
    request_body = {
        "song_title": "sweater weather",
        "artist_name": None,
        "top_n": TOP_N
    }
    response = client.post("/recommend", json=request_body)

    recommendations = response.json()["recommendations"]

    assert isinstance(recommendations, list)
    assert len(recommendations) == TOP_N

    for item in recommendations:
        assert isinstance(item["track_name"], str)
        assert isinstance(item["artist_name"], str)
        assert isinstance(item["genre"], str)
        assert isinstance(item["similarity"], float)
