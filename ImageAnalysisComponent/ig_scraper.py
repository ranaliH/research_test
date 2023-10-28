import instaloader
from azure.storage.blob import BlobServiceClient
import requests
import datetime

class GetInstagramImages():
    
    def __init__(self, connection_string, container_name) -> None:
        self.L = instaloader.Instaloader()
        self.connection_string = connection_string
        self.container_name = container_name

    def save_image_to_blob(self, image_url, image_name):
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(self.container_name)

        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            blob_client = container_client.get_blob_client(image_name)
            blob_client.upload_blob(response.content, overwrite=True)
            #print(f"Uploaded image {image_name} to Azure Blob Storage.")
        else:
            print(f"Error while downloading image: HTTP status code {response.status_code}")

    def scrape_and_store_images(self, username):
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
            posts = profile.get_posts()
            for post in posts:
                image_url = post.url
                image_name = f"{username}_{post.date_utc.strftime('%Y%m%d%H%M%S')}.jpg"
                self.save_image_to_blob(image_url, image_name)
        except instaloader.exceptions.ProfileNotExistsException as e:
            print(f"Profile '{username}' does not exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
      

    # def scrape_and_store_images(self, username):
    #     try:
    #         profile = instaloader.Profile.from_username(self.L.context, username)
    #         posts = profile.get_posts()
    #         current_date = datetime.datetime.now().date()
            
    #         for post in posts:
    #             post_date = post.date_utc.date()
                
    #             # Check if the post date is the same as the current date
    #             if post_date == current_date:
    #                 image_url = post.url
    #                 image_name = f"{username}_{post.date_utc.strftime('%Y%m%d%H%M%S')}.jpg"
    #                 self.save_image_to_blob(image_url, image_name)
                    
    #     except instaloader.exceptions.ProfileNotExistsException as e:
    #         print(f"Profile '{username}' does not exist.")
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")



