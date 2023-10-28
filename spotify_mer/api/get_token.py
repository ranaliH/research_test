import spotipy.util as util

# Set up your Spotify API credentials
client_id = 'your_client_id'
client_secret = 'your_client_secret'
redirect_uri = 'your_redirect_uri'

# Set your Spotify username and password
username = 'puranja@gmail.com'
password = 'ppw1999jp'

# Get the token using username and password
token = util.prompt_for_user_token( username=username, password=password)

# Use the token to make API requests
if token:
    # Your code to make API requests goes here
    print("Access token:", token)
else:
    print("Unable to obtain token")