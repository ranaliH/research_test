import requests
import json

access_token = 'BQAOnL-ovJ4_AljwTsE6UwU5ReBXBpw2bb1beZ3pxizb6olN5q64LAT4dFmp4VM8NrvC2WTkW4ainp575aJlUffBbDNshdqHheNHjojG6pUpNRtjcxqNBFuX3VOc8xRhlDuJy82LLKc7D1S6RPNMGzuzm4o8kXkwFdG_2-dT20_ryVYJNDN2lkh9EYYtT2rx0PkaiZXLerlBFgW4px6xA7YQpQIBKmiRUyzljOhPL_hfLCUtNEW-eNd83hyZnjws580muPQ1GP0nwsynAYln1_ZR'

headers = {
    "Authorization": f"Bearer {access_token}"
}

# Example song ID
song_id = "3y4LxiYMgDl4RethdzpmNe"

url = f"https://api.spotify.com/v1/audio-features/{song_id}"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    response_data = json.loads(response.content)
    # Access the song features
    danceability = response_data["danceability"]
    energy = response_data["energy"]
    # ... access other features as needed
    print("Danceability:", danceability)
    print("Energy:", energy)
    # ... print other features as needed
else:
    print("Error: could not get song features")
