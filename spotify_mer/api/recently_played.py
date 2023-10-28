from spotipy.oauth2 import SpotifyOAuth
import aiohttp
import asyncio
import nest_asyncio

# Authorization token that must have been created previously. See : https://developer.spotify.com/documentation/web-api/concepts/authorization
# token = 'BQAFpdS1c_tmPXBav0-FfV_q_6CGhjxc6gRo2Q_ETIA4ao3QNdtF5Ct786ou_XhXA1OkCx0QPBDKgDBRc0ADJHAlTxE7DxCurWFbmo1Cf8rKRzq40vz6w-MaiOZXPep059BUdhAmHDLCV0h-InAbAYaS-H9aBopgU9ANGKaUh4k9eZzJU344RSY40O0gh-VgmYfmANJoOi-T_0ugZJq1uwnYyRufEN-yGe9GLK9F0uySOjKlLoe9paUqsXUiMX4ge74TmpLqzD45_8ONwXOzbCV2'


def recently_played(token):
    nest_asyncio.apply()

    async def fetchWebApi(session, endpoint, method, body=None):
        async with session.request(method, f'https://api.spotify.com/{endpoint}', headers={'Authorization': f'Bearer {token}'}, json=body) as res:
            return await res.json()

    async def getTopTracks():
        async with aiohttp.ClientSession() as session:
            response = await fetchWebApi(session, 'v1/me/top/tracks?time_range=short_term&limit=10', 'GET')
            # response = await fetchWebApi(session, 'v1/me/player/recently-played?limit=10', 'GET')
            return response.get('items')

    loop = asyncio.get_event_loop()
    topTracks = loop.run_until_complete(getTopTracks())

    print([f"{track['name']} by {', '.join([artist['name'] for artist in track['artists']])}" for track in topTracks])

    topTrackIDs = [track['id'] for track in topTracks]
    return {'track_ids': topTrackIDs, 'tracks': [f"{track['name']} by {', '.join([artist['name'] for artist in track['artists']])}" for track in topTracks]}
