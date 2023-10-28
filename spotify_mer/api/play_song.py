import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888/callback"


# Set up authentication and initialize the Spotipy client
scope = "user-modify-playback-state"
client_id = "36084360a9524d9db1f07322ded21a90"
client_secret = "5a09cd42dbc8473eb7f46f476b2a9317"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id=client_id, client_secret=client_secret))
# Define the song's URI (e.g., "spotify:track:xxxxxxxxxxxx")
song_uri = "spotify:track:3Z28knVlVDHPkVU6X1fXfE"

# Start or resume the user's playback session
sp.start_playback(uris=[song_uri])