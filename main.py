from TextAnalysisComponent.prediction import get_most_predicted_class
from spotify_mer.api.recently_played import recently_played
from spotify_mer.feature_analysis.feature_analysis import get_feat_quad
from spotify_mer.utils.overall_quadrant import get_overall_quadrant
from ImageAnalysisComponent.ig_prediction import img_predictions
from ImageAnalysisComponent.ig_scraper import GetInstagramImages
import subprocess
import sys
import argparse

# set input parameters
# username = 'IzyaanImthiaz'
spotify_token = "BQBIUroPWQVemvkhLiwpo6-BCSsnhq-ROaPsnfzFZtCkD8La8-EW8B6YTKIxWc5i8t1QRX8dXy37NRkWthDHz5R1_guS2EjWwkk0K6qvcJ9Su02T58ulZ62pab7d29bLuPOTr3inGDSZStORG7PRicPDRii_xD6jkDbt3DzvTs2lumIHioHtj3It2m1zkVtyQ1PQmWClBN6WX-FPR0ulV7P82IRo5dIAo48VkDjU2JbRkFiDTh6NpPfkI8vKFwlxmBqEW9FX9PJVvAvncrcm4UBP"
connection_string = "DefaultEndpointsProtocol=https;AccountName=scrapeddataforapp;AccountKey=PlqU9/MzDy5yF9Si4xhLoGTk7jTg1XkR2V0IAQOSPkR+JTDYz1VByxUcSqd/WAj/ZW8wI9SDiZnv+ASt2IxVsw==;EndpointSuffix=core.windows.net"
container_name = "igscrapedata"
# igusername = "tinki7001"

def main():
    parser = argparse.ArgumentParser(description='Main Script')
    parser.add_argument('--name', type=str)
    parser.add_argument('--rdusername', type=str)
    parser.add_argument('--inusername', type=str)
    parser.add_argument('--spusername', type=str)
    parser.add_argument('--inpassword', type=str)
    args = parser.parse_args()

    # Extract arguments
    name = args.name
    rdusername = args.rdusername
    inusername = args.inusername
    spusername = args.spusername
    inpassword = args.inpassword
#text analysis
def text_analysis(username):
    predicted_class = get_most_predicted_class(username)
    return predicted_class
    # print('Predicted class for', username, ':', predicted_class)

#music analysis
def music_analysis(spotify_token):
    ids = recently_played(spotify_token)
    quadrants = get_feat_quad(ids)
    get_overall_quadrant(quadrants)

#image analysis
def image_analysis(connection_string,container_name,igusername):
    cls = GetInstagramImages(connection_string, container_name)
    cls.scrape_and_store_images(igusername)
    img_predictions()


def final_result(text_prediciton,music_prediction,image_preditction):
    final_result = 0
    if image_preditction != 'no stress' or text_prediciton != 'None':
        final_result=1
    return final_result

def run_detection(name, rdusername, inusername, spusername, inpassword):
    text_prediction = text_analysis(rdusername)
    print(text_prediction)
    image__prediction = image_analysis(connection_string,container_name,inusername)
    print(image__prediction)
    music_prediction = music_analysis(spotify_token)
    # music_prediction = 'music_analysis(spotify_token)'

    final_emotion = final_result(text_prediction,music_prediction,image__prediction)
    print(final_emotion)
    return final_emotion
# if __name__ == '__main__':
#     run_detection()
# if final_emotion == 1:
#     subprocess.run(["python", "Anupa/chatbot-deployment/app.py"], shell=True)