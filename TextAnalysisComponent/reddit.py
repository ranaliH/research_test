import praw
import pandas as pd

def reddit_connect():
    reddit = praw.Reddit(
        client_id="pF81TXSg5SAbElBXINx95w",
        client_secret="IhrOExP4Cf4YNGW6yTnfI_z_HJxzBg",
        user_agent="my user agent",
        username="",
        password="",
    )
    print(reddit.read_only)
    return reddit

red= reddit_connect()

# import praw
# import pandas as pd
# import warnings

# def reddit_connect():
#     reddit = praw.Reddit(
#         client_id="pF81TXSg5SAbElBXINx95w",
#         client_secret="IhrOExP4Cf4YNGW6yTnfI_z_HJxzBg",
#         user_agent="my user agent",
#         username="",
#         password="",
#     )
#     print(reddit.read_only)
#     return reddit

# # Suppress all PRAW warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="praw")

# red = reddit_connect()


def retrieve_user_data(username, reddit=red):
    data = []
    subreddits = ['depression', 'Anxiety', 'autism', 'bipolar', 'BPD', 'schizophrenia', 'mentalhealth', 'suicidewatch', 'ADHD', 'OCD', 'PTSD', 'EatingDisorders', 'BorderlinePDisorder', 'socialanxiety', 'selfharm', 'panicdisorder', 'insomnia', 'selfhelp', 'psychology', 'therapy', 'addiction', 'stress']

    for subreddit in subreddits:
        user = reddit.redditor(username)
        submissions = user.submissions.new()
        comments = user.comments.new()

        for submission in submissions:
            if submission.subreddit.display_name == subreddit:
                content = submission.title + '. ' + submission.selftext
                data.append({
                    'subreddit': subreddit,
                    'content': content,
                    'created_utc': submission.created_utc,
                    'id': submission.id,
                    'kind': 't3'
                })

        for comment in comments:
            if comment.subreddit.display_name == subreddit:
                content = comment.body
                data.append({
                    'subreddit': subreddit,
                    'content': content,
                    'link_id': comment.link_id,
                    'created_utc': comment.created_utc,
                    'id': comment.id,
                    'kind': 't1'
                })

    df = pd.DataFrame(data)
    return df



# def retrieve_user_data(username, reddit=red):
    # data = []
    # subreddits = ['depression', 'Anxiety', 'autism']
                  # #, 'bipolar', 'BPD', 'autism', 'schizophrenia', 'mentalhealth', 'suicidewatch', 'ADHD', 'OCD', 'PTSD', 'EatingDisorders', 'BorderlinePDisorder', 'socialanxiety', 'selfharm', 'panicdisorder', 'insomnia', 'selfhelp', 'psychology', 'therapy', 'addiction', 'stress']

    # for subreddit in subreddits:
        # user = reddit.redditor(username)
        # submissions = user.submissions.new(limit=10)
        # comments = user.comments.new(limit=10)

        # for submission in submissions:
            # if submission.subreddit.display_name == subreddit:
                # data.append({
                    # 'subreddit': subreddit,
                    # 'content': submission.title,
                    # 'created_utc': submission.created_utc,
                    # 'id': submission.id,
                    # 'kind': 't3'
                # })

        # for comment in comments:
            # if comment.subreddit.display_name == subreddit:
                # data.append({
                    # 'subreddit': subreddit,
                    # 'content': comment.body,
                    # 'link_id': comment.link_id,
                    # 'created_utc': comment.created_utc,
                    # 'id': comment.id,
                    # 'kind': 't1'
                # })

    # return data


# # red = reddit_connect()
# username = "IzyaanImthiaz"

# data = retrieve_user_data(username)
# for item in data:
#     print(item['content'])



# import requests
# import praw

# username = "IzyaanImthiaz"
# password = "!@#$QWER"
# user_agent = "MyAPI/0.0.1"



# # Reddit API credentials

# CLIENT_ID ='pF81TXSg5SAbElBXINx95w'
# SECRET_TOKEN ='IhrOExP4Cf4YNGW6yTnfI_z_HJxzBg'
# REDIRECT_URI = 'http://localhost:8080'


# def redditConnect():
#     reddit = praw.Reddit(
#         client_id="pF81TXSg5SAbElBXINx95w",
#         client_secret="IhrOExP4Cf4YNGW6yTnfI_z_HJxzBg",
#         user_agent="my user agent",
#         username="",
#         password="",
#     )
#     print(reddit.read_only)
#     return reddit

# red= redditConnect()

# def printPostFromSubreddit(redditCon, subreddit):
#     for submission in redditCon.subreddit(subreddit).hot(limit=10):
#         print(submission.title)

# printPostFromSubreddit(red,'depression')

# import praw
# import pandas as pd


# # Define the mental health subreddits
# mental_health_subreddits = ['depression', 'Anxiety', 'bipolar', 'BPD', 'autism', 'schizophrenia', 'mentalhealth',
#                             'suicidewatch', 'ADHD', 'OCD', 'PTSD', 'EatingDisorders', 'BorderlinePDisorder',
#                             'socialanxiety', 'selfharm', 'panicdisorder', 'insomnia', 'selfhelp', 'psychology',
#                             'therapy', 'addiction', 'stress']

# def retrieve_user_data(username):
#     reddit = praw.Reddit(
#         client_id="YOUR_CLIENT_ID",
#         client_secret="YOUR_CLIENT_SECRET",
#         user_agent="YOUR_USER_AGENT",
#     )

#     user_data = []

#     for subreddit in mental_health_subreddits:
#         subreddit_instance = reddit.subreddit(subreddit)

#         for submission in subreddit_instance.new(limit=10):
#             if submission.author and submission.author.name == username:
#                 user_data.append({
#                     'subreddit': subreddit,
#                     'content': submission.title,
#                     'created_utc': submission.created_utc,
#                     'id': submission.id,
#                     'kind': 't3'
#                 })

#             submission.comments.replace_more(limit=None)
#             for comment in submission.comments.list():
#                 if comment.author and comment.author.name == username:
#                     user_data.append({
#                         'subreddit': subreddit,
#                         'content': comment.body,
#                         'link_id': comment.link_id,
#                         'created_utc': comment.created_utc,
#                         'id': comment.id,
#                         'kind': 't1'
#                     })

#     return pd.DataFrame(user_data)

# # Call the function to retrieve user data
# username = input("Enter the username: ")
# user_data_df = retrieve_user_data(username)

# # Print the DataFrame
# print(user_data_df)

# import asyncpraw
# import pandas as pd
# from datetime import datetime

# # Authenticate with the Reddit API using Async PRAW
# async def reddit_connect():
#     reddit = await asyncpraw.reddit.AsyncReddit(
#         client_id="YOUR_CLIENT_ID",
#         client_secret="YOUR_CLIENT_SECRET",
#         user_agent="YOUR_USER_AGENT",
#         username="YOUR_USERNAME",
#         password="YOUR_PASSWORD",
#     )
#     print(reddit.read_only)
#     return reddit

# # Retrieve posts and comments from a user in specific subreddits
# async def retrieve_user_data(reddit, username, subreddits):
#     data = []

#     for subreddit in subreddits:
#         async for submission in reddit.redditor(username).submissions.new(limit=10):
#             if submission.subreddit.display_name == subreddit:
#                 created_utc = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%dT%H:%M:%SZ')

#                 data.append({
#                     'subreddit': subreddit,
#                     'content': submission.title,
#                     'created_utc': created_utc,
#                     'id': submission.id,
#                     'kind': 't3'
#                 })

#         async for comment in reddit.redditor(username).comments.new(limit=10):
#             if comment.subreddit.display_name == subreddit:
#                 created_utc = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%dT%H:%M:%SZ')

#                 data.append({
#                     'subreddit': subreddit,
#                     'content': comment.body,
#                     'link_id': comment.link_id,
#                     'created_utc': created_utc,
#                     'id': comment.id,
#                     'kind': 't1'
#                 })

#     df = pd.DataFrame(data)
#     return df

