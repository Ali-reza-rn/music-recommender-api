"""Tests for the /search endpoint.

Covers prefix matching, unknown songs and artists (404), empty input (422),
and case-insensitive matching. The `client` fixture comes from conftest.py.
"""

import pytest


@pytest.mark.parametrize("request_body, expected_song_title, expected_artist",
    [
    (
        {
            "song_title": "sweater",
            "artist_name": None
        },
        "sweater weather", "the neighbourhood"
    ),
    (
        {
            "song_title": "love",
            "artist_name": "selena"
        },
        "love you like a love song", "selena gomez & the scene"
    ),
    (
        {
            "song_title": "shape",
            "artist_name": "sti"
        }, "shape of my heart", "sting"
    )
])
def test_search_returns_correct_song(
    client, request_body, expected_song_title, expected_artist
    ):
    response = client.get("/search", params=request_body)
    assert response.status_code == 200
    body = response.json()[0]
    assert body["track_name"] == expected_song_title
    assert body["artist_name"] == expected_artist


@pytest.mark.parametrize("request_body, expected_code", [
    (
        {
        "song_title": "there is no song with this name in dataset",
        "artist_name": "this a wrong artist_name",
        }, 404
    ),
    (
        {
        "song_title": "shape of my heart",
        "artist_name": "this a wrong artist_name"
        }, 404
    ),
    (
        {
            "song_title": "there is no song with this name in dataset",
            "artist_name": "sting"
        }, 404
    ),
    ({"song_title": "", "artist_name": ""}, 422)
], ids=[
    "Wrong song_title and artist_name",
    "Correct song_title and wrong artist_name",
    "Correct artist_name and wrong song_title",
    "Empty song_title and artist_name"
])
def test_search_rejects_invalid_input(client, request_body, expected_code):
    response = client.get("/search", params=request_body)
    assert response.status_code == expected_code


# Case Insensitive
@pytest.mark.parametrize("request_body", [
    {"song_title": "SwEaTeR WeAtHeR", "artist_name": None},
    {"song_title": "SwEaTeR WeAtHeR", "artist_name": "tHe neiGhbouRhOOd"}
])
def test_search_is_case_insensitive(client, request_body):
    response = client.get("/search", params=request_body)
    assert response.status_code == 200
