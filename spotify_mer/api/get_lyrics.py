import lyricsgenius as genius
import pandas as pd

genius_token = 'v5_uCYZLxt7NqfhAlpmac3g1TGvIgy6jPr2oY2695nv7GI9emvUjhp4USu2MxhrP'

def get_lyrics(song,artist):

    api=genius.Genius(genius_token)
    song = api.search_song('Bohemian Rhapsody', 'Queen')
    lyrics = song.lyrics

    # with open('lyrics.txt', 'w') as f:
    #     f.write(lyrics)
    lines = lyrics.split('\n')

    # Create a DataFrame with the lyrics lines
    df = pd.DataFrame({'Lyrics': lines})
    print('reached end')

get_lyrics('Bohemian Rhapsody', 'Queen')
a=0
    # Print the lyrics
    # print(lyrics)