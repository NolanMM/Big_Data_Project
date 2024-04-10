# Databricks notebook source
# MAGIC %md
# MAGIC # YouTube Data Analytics and Sentiment Analysis
# MAGIC
# MAGIC **Objective**: The goal of this project is to integrate big data workflows, including data extraction,
# MAGIC analysis, and visualization, using YouTube as a data source. You will explore various aspects of
# MAGIC YouTube analytics, such as view trends, content topics, and viewer sentiment, to gain insights into
# MAGIC content strategy and audience engagement.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import necessary libraries

# COMMAND ----------

!pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
!pip install textblob
!pip install wordcloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
import matplotlib.ticker as ticker
from textblob import TextBlob
from wordcloud import WordCloud

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Extraction and Initial Analysis
# MAGIC **a)** Execute the YouTube API request to retrieve video ID, title, publish time, view count, and
# MAGIC comment count.
# MAGIC
# MAGIC **b)** A well-structured dataset containing the fetched data.

# COMMAND ----------

API_KEY = 'AIzaSyDJYDUz82ALvQlNSCVJaRvWnTYKHBg3HzU'
channel = 'UC8butISFwT-Wl7EV0hUK0BQ'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_channel_stats(youtube, channel_id):
    """
    Retrieves detailed statistics and metadata of a YouTube channel using the YouTube Data API.

    Parameters:
        youtube (googleapiclient.discovery.Resource): An authorized resource object for interacting with the YouTube Data API.
        channel_id (str): The unique identifier of the YouTube channel.

    Returns:
        dict: A dictionary containing comprehensive statistics and metadata of the specified YouTube channel.

        # Initialize the YouTube Data API client
        youtube = build('youtube', 'v3', developerKey='YOUR_API_KEY')

        # Retrieve statistics for a YouTube channel with the specified channel ID
        channel_statistics = get_channel_stats(youtube, 'UC_x5XG1OV2P6uZZ5FSM9Ttw')

        # Output the retrieved statistics
        print(channel_statistics)
    """
    request = youtube.channels().list(part='snippet,contentDetails,statistics', id=channel_id)
    response = request.execute()
    return response['items']

def get_video_list(youtube,  playlist_ID):
    """
    Retrieves a list of video IDs from a YouTube playlist using the YouTube Data API.

    Parameters:
        youtube (googleapiclient.discovery.Resource): An authorized resource object for interacting with the YouTube Data API.
        playlist_ID (str): The unique identifier of the YouTube playlist.

    Returns:
        list: A list of video IDs contained within the specified playlist.

        # Initialize the YouTube Data API client
        youtube = build('youtube', 'v3', developerKey='YOUR_API_KEY')

        # Retrieve the list of video IDs from a YouTube playlist with the specified playlist ID
        video_list = get_video_list(youtube, 'PLwJS-GTzxiESBmJ4gJj-1wOuP5JoE41h4')

        # Output the retrieved list of video IDs
        print(video_list)
    """
    video_list = []
    request = youtube.playlistItems().list(
        part = "snippet,contentDetails",
        playlistId = playlist_ID,
        maxResults = 50
    )
    next_page = True
    while next_page:
        response = request.execute()
        data = response['items']

        for video in data:
            video_id = video['contentDetails']['videoId']
            if video_id not in video_list:
                video_list.append(video_id)

        if 'nextPageToken' in response:
            next_page = True
            request = youtube.playlistItems().list(
                part = "snippet,contentDetails",
                playlistId = playlist_ID,
                maxResults = 50,
                pageToken = response['nextPageToken']
            )
        else:
            next_page = False
    
    return video_list

def get_video_details(youtube, video_list):
    """
    Retrieves detailed statistics and metadata of YouTube videos using the YouTube Data API.

    Parameters:
        youtube (googleapiclient.discovery.Resource): An authorized resource object for interacting with the YouTube Data API.
        video_list (list): A list of video IDs whose details need to be fetched.

    Returns:
        list: A list of dictionaries, each containing comprehensive statistics and metadata of a YouTube video.

        # Initialize the YouTube Data API client
        youtube = build('youtube', 'v3', developerKey='YOUR_API_KEY')

        # Retrieve statistics for a list of YouTube videos with the specified video IDs
        video_data = get_video_details(youtube, ['VIDEO_ID_1', 'VIDEO_ID_2'])

        # Output the retrieved video statistics
        for video in video_data:
            print(video)
    """
    stats_list = []

    for i in range(0, len(video_list), 50):
        request = youtube.videos().list(
            part = "snippet,contentDetails,statistics",
            id = video_list[i:i+50]
        )
        response = request.execute()
        
        for video in response['items']:
            video_id = video['id']
            title = video['snippet']['title']
            published = video['snippet']['publishedAt']
            description = video['snippet']['description']
            tag_count = len(video['snippet'].get('tags',[]))
            views_count = video['statistics'].get('viewCount',0)
            dislikes_count = video['statistics'].get('dislikeCount',0)
            likes_count = video['statistics'].get('likeCount',0)
            comments_count = video['statistics'].get('commentCount',0)

            stats_dictionary = dict(
                video_id= video_id,
                title=title,
                published=published,
                description=description,
                tag_count=tag_count,
                views_count=views_count,
                dislikes_count=dislikes_count,
                likes_count=likes_count,
                comments_count=comments_count
            )

            stats_list.append(stats_dictionary)

    return stats_list

def data_extraction_process(youtube=None, channel=None):
    """
    Performs the process of extracting data from a YouTube channel, creates a DataFrame, and processes the data.

    Parameters:
        youtube (googleapiclient.discovery.Resource, optional): An authorized resource object for interacting with the YouTube Data API.
            Defaults to None.
        channel (str, optional): The unique identifier of the YouTube channel.
            Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted video data and processed metrics.
    """
    if youtube is None or channel is None:
        raise ValueError("'youtube' and 'channel' parameters are required.")

    channel_stats = get_channel_stats(youtube, channel)
    playlist_ID_List = channel_stats[0]['contentDetails']['relatedPlaylists']['uploads']
    video_list = get_video_list(youtube, playlist_ID_List)
    video_data = get_video_details(youtube, video_list)
    
    # Convert to pandas dataframe and Data enrichment
    df = pd.DataFrame(video_data)
    df['title_length'] = df['title'].str.len()
    df['views_count'] = pd.to_numeric(df['views_count'])
    df['published'] = pd.to_datetime(df['published'])
    df['likes_count'] = pd.to_numeric(df['likes_count'])
    df['dislikes_count'] = pd.to_numeric(df['dislikes_count'])
    df['comments_count'] = pd.to_numeric(df['comments_count'])
    df['reactions'] = df['likes_count'] + df['dislikes_count'] + df['comments_count']
    df['publish_month'] = df['published'].dt.month
    df['publish_day'] = df['published'].dt.day
    df['publish_year'] = df['published'].dt.year
    df['publish_hour'] = df['published'].dt.hour
    df['publish_period'] = df['publish_hour'].apply(lambda x: 'AM' if x < 12 else 'PM')

    return df

# COMMAND ----------

df = data_extraction_process(youtube, channel)
df.describe()

# COMMAND ----------

display(df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. EDA Process (Exploring Data Analysis) 
# MAGIC - **View Trend Analysis**
# MAGIC - **Publishing Frequency Analysis**
# MAGIC
# MAGIC **a)** Using Time Series Analysis method and visualize the data to analyze the trend of view over the time
# MAGIC
# MAGIC **b)** Plot the number of videos published each month to analyze content frequency to helps
# MAGIC understand the channel's consistency and its potential impact on subscriber engagement and
# MAGIC channel growth.
# MAGIC
# MAGIC **c)** Deliverable: A time-series plot and analysis of publishing patterns.

# COMMAND ----------

df_sort_by_time = df.sort_values(by='published')

def millions_formatter(x, pos):
    """
    The millions_formatter function is a custom formatter used to format y-axis labels in millions for matplotlib plots. It takes two parameters: x (the tick value) and pos (the tick position), and returns a string representing the formatted label.

    Parameters:
        x (float): The tick value.
        pos (int): The tick position.

    Returns:
        str: A string representing the formatted label, with the value divided by 1e6 (1 million) and rounded to 0 decimal places, followed by the letter 'M' indicating millions.

    Example usage:
        # Define a formatter function for millions
        def millions_formatter(x, pos):
            return '{:.0f}M'.format(x / 1e6)

        # Set y-axis labels to display values in millions
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
    """
    return '{:.0f}M'.format(x / 1e6)

# COMMAND ----------

def summarize_view_trend_analysis(df_sort_by_time = df_sort_by_time):
    """
    The summarize_view_trend_analysis function generates visual summaries of the trend analysis for total views over different time periods such as years, months, periods within a day, and hours. It groups the provided DataFrame by various time-related columns and calculates the total views for each group. It then creates plots to visualize these trends using matplotlib and seaborn.

    Parameters:
        df_sort_by_time (pandas.DataFrame, optional): The DataFrame containing sorted data based on the 'published' column. Defaults to the value of df_sort_by_time.

    Visualization Details:
        - Trend of Total Views Over Year: This line plot illustrates the trend of total views over years.
        - Trend of Total Views Over Months: This line plot illustrates the trend of total views over months.
        - Total Views by Period in Day: This bar plot displays the total views categorized by periods within a day.
        - Total Views by Hour: This line plot shows the total views across different hours of the day.
    """
    df_year_total_views = df_sort_by_time.groupby('publish_year')['views_count'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(df_year_total_views['publish_year'], df_year_total_views['views_count'], marker='o', linestyle='-',color='b')
    plt.title('Trend of Total Views Over Year')
    plt.xlabel('Year')
    plt.ylabel('Total Views Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
    plt.tight_layout()
    plt.show()

    df_month_total_views = df_sort_by_time.groupby('publish_month')['views_count'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(df_month_total_views['publish_month'], df_month_total_views['views_count'], marker='o', linestyle='-',color='b')
    plt.title('Trend of Total Views Over Months')
    plt.xlabel('Month')
    plt.ylabel('Total Views Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
    plt.tight_layout()
    plt.show()

    df_period_in_day_total_views = df_sort_by_time.groupby('publish_period')['views_count'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(df_period_in_day_total_views['publish_period'], df_period_in_day_total_views['views_count'], color='b')
    plt.xlabel('Period')
    plt.ylabel('Total Views')
    plt.title('Total Views by Period in Day')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
    plt.tight_layout()
    plt.show()

    df_hour_in_day_total_views = df_sort_by_time.groupby('publish_hour')['views_count'].sum().reset_index()

    sns.set(rc={'figure.figsize':(10,8)})
    plot = sns.lineplot(data=df_hour_in_day_total_views, x="publish_hour", y="views_count", marker="o", markersize=10,color='b')
    plot.set(xlabel='Hour', ylabel='Total Views')
    plot.set_title('Total Views by Hour')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
    plt.tight_layout()
    plt.show()

summarize_view_trend_analysis()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sentiment Analysis of Video Titles
# MAGIC **a)** Perform sentiment analysis on video titles using NLP TextBlob Python Library to categorize
# MAGIC them into Positive, Neutral, and Negative.
# MAGIC
# MAGIC **b)** Deliverable: A pie chart showing the sentiment distribution and an analysis of content strategy
# MAGIC or viewer engagement.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Calculate sentiment

# COMMAND ----------

def get_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Get the sentiment polarity
    sentiment_polarity = blob.sentiment.polarity
    # Determine sentiment category based on polarity
    sentiment_category = 'Positive' if sentiment_polarity > 0 else 'Neutral' if sentiment_polarity == 0 else 'Negative'
    return sentiment_category, sentiment_polarity

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Data Aggregation and Summary Statistics

# COMMAND ----------

def data_aggregation_and_summary_statistics(df = df):
  # Apply the function to each video title or comment text
  df['sentiment_category'], df['sentiment_polarity'] = zip(*df['title'].apply(get_sentiment))

  # Calculate the proportion of positive, negative, and neutral titles
  sentiment_proportions = df['sentiment_category'].value_counts(normalize=True)

  # Display the sentiment proportions
  print("Sentiment Proportions:")
  print(sentiment_proportions)

  # Extract just the date part for easier correlation analysis
  df['publish_date'] = df['published'].dt.date

  # Calculate the mean view count for each sentiment category
  mean_view_counts = df.groupby('sentiment_category')['views_count'].mean()

  # Display the mean view counts by sentiment
  print("\nMean View Counts by Sentiment:")
  print(mean_view_counts)

  videos_df = df

  # Convert sentiment to a numerical scale for correlation analysis
  videos_df['sentiment_score'] = videos_df['sentiment_polarity'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

  # Calculate correlation of sentiment with view count and publish time
  sentiment_view_correlation = videos_df['sentiment_score'].corr(videos_df['views_count'])
  sentiment_time_correlation = videos_df['sentiment_score'].corr(videos_df['published'].astype(int))

  # Display the correlation results
  print("\nCorrelation between Sentiment and View Count:")
  print(sentiment_view_correlation)
  print("\nCorrelation between Sentiment and Publish Time:")
  print(sentiment_time_correlation)

  return videos_df

videos_df_raw = data_aggregation_and_summary_statistics()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3 Visualize the statistics data
# MAGIC - Sentiment Distribution of Video Titles
# MAGIC - Sentiment Trend Over Time
# MAGIC - Number of Videos per Sentiment Category

# COMMAND ----------

def visualize_statistic_distribution_data(videos_df = data_aggregation_and_summary_statistics()):
    # Sentiment distribution pie chart
    plt.figure(figsize=(7, 7))
    videos_df['sentiment_category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Exercise 2.5 (PieChart)- Sentiment Distribution of Video Titles')
    plt.ylabel('')  # Hide the 'sentiment_category' label on the y-axis
    plt.show()

    # Time series plot of sentiment over time
    plt.figure(figsize=(12, 6))
    videos_df['publish_date'] = pd.to_datetime(videos_df['publish_date'])
    videos_df.sort_values('publish_date', inplace=True)
    videos_df['numeric_sentiment'] = videos_df['sentiment_polarity'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    videos_df.groupby('publish_date')['numeric_sentiment'].mean().plot()
    plt.title('Exercise 2.5 (TimeSeriesPlot)- Sentiment Trend Over Time')
    plt.xlabel('Publish Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()

    # Bar graph of sentiment categories
    plt.figure(figsize=(10, 6))
    videos_df['sentiment_category'].value_counts().plot.bar(color=['green', 'blue', 'red'])
    plt.title('Exercise 2.5 (Barchart)- Number of Videos per Sentiment Category')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=0)
    plt.show()

# COMMAND ----------

def _4_sentiment_analysis_of_video_main_process():
    data_aggregation_and_summary_statistics()
    visualize_statistic_distribution_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. In-depth Sentiment Analysis of Comments
# MAGIC **a)** Identify the top 5 videos with the maximum comment count and display them.
# MAGIC
# MAGIC **b)** Select the video with the maximum comment count and perform detailed sentiment analysis
# MAGIC on its comments to understand public opinion.
# MAGIC
# MAGIC **c)** Fetch the comments, analyze the sentiments, and categorize them into Positive, Negative, and
# MAGIC Neutral. Create and display the pie chart for sentiment distribution.
# MAGIC
# MAGIC **d)** Create and display a word cloud for each sentiment category.
# MAGIC
# MAGIC **e)** Deliverable
# MAGIC - Pie chart for sentiment distribution.
# MAGIC - word cloud for each sentiment category.
# MAGIC - comprehensive insights for each analysis step.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1 Identify the top 5 videos with the maximum comment count

# COMMAND ----------

def identify_top_5_videos_by_count(videos_df = data_aggregation_and_summary_statistics()):
    #  Sort the DataFrame based on the 'comment_count' column in descending order
    videos_df_sorted = videos_df.sort_values(by='comments_count', ascending=False)
    # Select the top 5 videos with the maximum comment count
    top_5_videos = videos_df_sorted.head(5)
    display(top_5_videos)
    return top_5_videos
    
identify_top_5_videos_by_count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 Analyze the video with the maximum comment count and perform detailed sentiment analysis on its comments to understand public opinion

# COMMAND ----------

def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        # Check if there are more pages
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments

def get_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Return the polarity
    return blob.sentiment.polarity

def categorize_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def analyze_sentiment_video_with_maximum_comment_count(top_videos = identify_top_5_videos_by_count()):
    # Fetch comments for a specific video
    video_id = top_videos.iloc[0]['video_id']
    comments = get_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')

    # Add sentiment analysis to the comments
    comments_sentiment = [{'comment': comment, 'sentiment_polarity': get_sentiment(comment)} for comment in comments]

    # Convert to DataFrame
    comments_df = pd.DataFrame(comments_sentiment)

    # Apply the categorization function
    comments_df['sentiment_category'] = comments_df['sentiment_polarity'].apply(categorize_sentiment)

    # Display the first few rows of the DataFrame
    display(comments_df)
    return comments_df, video_id

comments_df, video_id = analyze_sentiment_video_with_maximum_comment_count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.3 Fetch the comments, analyze the sentiments, and categorize them into Positive, Negative, and Neutral. Create and display the pie chart for sentiment distribution

# COMMAND ----------

def pie_chart_sentiment_category_distribution(comments_df = comments_df):
    sentiment_counts = comments_df['sentiment_category'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Exercise 2.7- Sentiment Category Distribution on Video Comments')
    plt.show()
    
pie_chart_sentiment_category_distribution()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4 Create and display a word cloud for each sentiment category

# COMMAND ----------

def display_word_cloud_by_sentiment_category(comments_df = comments_df):
  # Filter comments by sentiment category
  positive_comments = ' '.join(comments_df[comments_df['sentiment_category'] == 'Positive']['comment'])
  negative_comments = ' '.join(comments_df[comments_df['sentiment_category'] == 'Negative']['comment'])
  neutral_comments = ' '.join(comments_df[comments_df['sentiment_category'] == 'Neutral']['comment'])

  # Create a word cloud for each sentiment category
  wordclouds = {
      'Positive': WordCloud(width=800, height=400, background_color='white').generate(positive_comments),
      'Negative': WordCloud(width=800, height=400, background_color='white').generate(negative_comments),
      'Neutral': WordCloud(width=800, height=400, background_color='white').generate(neutral_comments)
  }

  # Display the word clouds for each sentiment category
  plt.figure(figsize=(15, 7.5))

  for i, (sentiment, wordcloud) in enumerate(wordclouds.items()):
      plt.subplot(1, 3, i+1)
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.title(f'{sentiment} Comments')
      plt.axis('off')

  plt.tight_layout()
  plt.show()

display_word_cloud_by_sentiment_category()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.5 Comparison and Insights

# COMMAND ----------

def comparison_and_insights(videos_df = videos_df_raw, comments_df = comments_df, video_id = video_id):
    # Calculate the average sentiment score for the video title
    video_sentiment_score = videos_df.loc[videos_df['video_id'] == video_id, 'sentiment_polarity'].mean()

    # Calculate the average sentiment score for the comments
    comments_sentiment_score = comments_df['sentiment_polarity'].mean()

    # Display the average sentiment scores
    print(f"Average sentiment score for the video title: {video_sentiment_score}")
    print(f"Average sentiment score for the comments: {comments_sentiment_score}")

    # Discuss the alignment or discrepancy
    if video_sentiment_score * comments_sentiment_score > 0:
        if abs(video_sentiment_score - comments_sentiment_score) < 0.1:
            print("The sentiments of the video title and the comments are closely aligned.")
        else:
            print("The sentiments of the video title and the comments are aligned but vary in intensity.")
    elif video_sentiment_score * comments_sentiment_score < 0:
        print("The sentiments of the video title and the comments are in discrepancy.")
    else:
        print("One of the sentiments is neutral, indicating mixed reactions.")

    # Further analysis can be done by looking at the distribution of sentiments
    print("\nSentiment Distribution in Video Comments:")
    print(comments_df['sentiment_category'].value_counts())

    # Compare with the sentiment of the video title
    video_sentiment_category = videos_df.loc[videos_df['video_id'] == video_id, 'sentiment_category'].iloc[0]
    print(f"\nSentiment of the video title: {video_sentiment_category}")

comparison_and_insights()
