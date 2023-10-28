import requests
from bs4 import BeautifulSoup

API_KEY = 'AIzaSyDyL60hFWMcch8ACJRt6uMe91uYQCmx1Mk'

def html_to_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    return text

def translate_text(text):
    url = 'https://translation.googleapis.com/language/translate/v2'
    params = {
        'key': API_KEY,
        'q': text,
        'target': 'si'
    }
    response = requests.post(url, params=params)
    translation = response.json()['data']['translations'][0]['translatedText']
    translation = html_to_text(translation)
    return translation


def translate_text_siToEn(text):
    url = 'https://translation.googleapis.com/language/translate/v2'
    params = {
        'key': API_KEY,
        'q': text,
        'target': 'en'
    }
    response = requests.post(url, params=params)
    translation = response.json()['data']['translations'][0]['translatedText']
    translation = html_to_text(translation)
    return translation
