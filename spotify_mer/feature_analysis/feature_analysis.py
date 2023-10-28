import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def get_valence(id):
    # Replace 'YOUR_CLIENT_ID' and 'YOUR_CLIENT_SECRET' with your actual client ID and client secret
    client_id = '36084360a9524d9db1f07322ded21a90'
    client_secret = '5a09cd42dbc8473eb7f46f476b2a9317'

    # Set up the client credentials manager
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

    # Create the Spotify client
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Specify the Spotify track ID
    # track_id = '4GCGH6TJ69neckwITeBFXK'

    # Get track features
    track_features = sp.audio_features(id)

    # Extract energy and valence from track features
    energy = track_features[0]['energy']
    valence = track_features[0]['valence']

    # Print the energy and valence values
    # print("Valence:", valence)
    return valence

def get_arousal(id):
    # Replace 'YOUR_CLIENT_ID' and 'YOUR_CLIENT_SECRET' with your actual client ID and client secret
    client_id = '36084360a9524d9db1f07322ded21a90'
    client_secret = '5a09cd42dbc8473eb7f46f476b2a9317'

    # Set up the client credentials manager
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

    # Create the Spotify client
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Specify the Spotify track ID
    # track_id = '4GCGH6TJ69neckwITeBFXK'

    # Get track features
    track_features = sp.audio_features(id)

    # Extract energy and valence from track features
    energy = track_features[0]['energy']

    # Print the energy and valence values
    # print("Energy:", energy)
    return energy


arousal_threshold = 0.5
valence_threshold = 0.5

def get_feat_quad(id):

    quadrants=[]
    arousals = []
    vals = []
    for i,id in enumerate(id['track_ids']):
        arousal = get_arousal(id)
        valence = get_valence(id)
        arousals.append(arousal)
        vals.append(valence)
        a=0

    for i in range(len(arousals)):
        if arousals[i] >= arousal_threshold and vals[i] >= valence_threshold:
            quadrant = 1 #happy
        elif arousals[i] >= arousal_threshold and vals[i] < valence_threshold:
            quadrant = 2 #angry
        elif arousals[i] < arousal_threshold and vals[i] >= valence_threshold:
            quadrant = 3 #tender
        else:
            quadrant = 4 #sad
    # if arousal >= arousal_threshold and valence >= valence_threshold:
    #     quadrant = 'Quadrant 1'
    # elif arousal >= arousal_threshold and valence < valence_threshold:
    #     quadrant = 'Quadrant 2'
    # elif arousal < arousal_threshold and valence >= valence_threshold:
    #     quadrant = 'Quadrant 3'
    # else:
    #     quadrant = 'Quadrant 4'
        quadrants.append(quadrant)
    print(quadrants)
    return quadrants

# get_feat_quad('4GCGH6TJ69neckwITeBFXK')