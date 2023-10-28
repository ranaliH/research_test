import requests
import base64

# Spotify API credentials
CLIENT_ID = '36084360a9524d9db1f07322ded21a90'
CLIENT_SECRET = '5a09cd42dbc8473eb7f46f476b2a9317'

# Base64 encoded credentials for authorization
credentials = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

def get_access_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f"Basic {credentials}"
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(auth_url, headers=headers, data=data)
    access_token = response.json().get('access_token')
    return access_token