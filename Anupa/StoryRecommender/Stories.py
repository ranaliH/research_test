from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import requests
from bs4 import BeautifulSoup as bs
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Anupa/SentimentAnalysis_Training'))
from Sentiment_Analysis1 import predict_sentiment

def scrape(Problem):
   
    query = Problem
    query = query.replace(' ', '+')
    url = 'https://www.goodnewsnetwork.org/?s=' + query
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    parent_div = driver.find_element(By.CLASS_NAME,'td-ss-main-content')

    
    nested_divs = parent_div.find_elements(By.XPATH,'.//div[2]//h3//a')
    links=[]
    for div in nested_divs[:5]:
        link = div.get_property('href')
        links.append(link)
    
    texts=[]
    for index in range(0, 5):
        req = requests.get(links[index])
        soup = bs(req.text, 'html.parser')

        parent = soup.find('div', attrs={'class': 'td-ss-main-content'})
        p_tags = parent.find_all('p')
        
        for p in p_tags:
            if p.text.strip():
                texts.append(p.text)
                break 

    result = {}
    for key, value in zip(links, texts):
        if key in result:
            result[key].append(value)
        else:
            result[key] = [value]
    
    driver.quit()

    # for key in my_dict:
    #     my_dict[key].append(new_value)
    for key, value in result.items():
        sentiment=predict_sentiment(value)
        if sentiment == 'positive':
            result[key].append(sentiment)
        
    hyperlinks = [link for link in result.keys()]
    return hyperlinks


# x=scrape("stress")
# print(x)