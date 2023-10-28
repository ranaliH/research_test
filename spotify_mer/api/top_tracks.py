from spotipy.oauth2 import SpotifyOAuth
import aiohttp
import asyncio

# Authorization token that must have been created previously. See : https://developer.spotify.com/documentation/web-api/concepts/authorization
token = 'BQBydWtVQzHcSjQJgPM3IQqUejQTGH0r7MbBaDDlfyzmjlXgj2T4QvQTlL1AigViWREfrLYCwPLts0ul9fgHwMDJ-vkWKN_ZBJC4buI3RKLJtCUgvC6hM25FVlTV_h3pQHlYMfBy7n2pkQf1tiKweuv-l4H2lX1Dzw0v-Lbt3ANOk65vVlKONxNj8p4DSqlC4AAj8Siq5VfHCApejq9DLRRPHBDKlXq3J4PEyIpeLv2Ja7L7ODXSeDoAESuJe69sXECxbApU_E9quMxuBpotey3H'

async def fetchWebApi(session, endpoint, method, body=None):
    # async with session.request(method, f'https://api.spotify.com/{endpoint}', headers={'Authorization': f'Bearer {token}'}, json=body) as res:
    
    async with session.request(method, f'https://api.spotify.com/{endpoint}', headers={'Authorization': f'Bearer {token}'}, json=body) as res:
        return await res.json()

async def getTopTracks():
    async with aiohttp.ClientSession() as session:
        response = await fetchWebApi(session, 'v1/me/top/tracks?time_range=long_term&limit=10', 'GET')
        # response = await fetchWebApi(session, 'v1/me/player/recently-played?limit=10', 'GET')
        return response.get('items')

loop = asyncio.get_event_loop()
topTracks = loop.run_until_complete(getTopTracks())

print([f"{track['name']} by {', '.join([artist['name'] for artist in track['artists']])}" for track in topTracks])
