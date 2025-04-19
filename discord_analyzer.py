import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
from collections import Counter
import seaborn as sns
from textblob import TextBlob

# Simple word tokenizer function to replace NLTK's word_tokenize
def simple_word_tokenize(text):
    """
    A simple word tokenizer that splits text on whitespace and punctuation
    """
    if not isinstance(text, str):
        return []
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace
    return [word for word in text.split() if word]

# Simple stopwords list (most common English stopwords)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 
    'won', 'wouldn'
])

def analyze_discord_chats(csv_file, output_dir):
    """
    Analyze Discord chat data from a CSV file
    
    Parameters:
    csv_file (str): Path to the CSV file containing Discord chat data
    output_dir (str): The directory to save visualizations to
    
    Returns:
    dict: A dictionary containing various analysis results
    """
    print(f"Analyzing Discord chat data from: {csv_file}")
    
    # Function to extract words from a message
    def extract_words(message):
        if not isinstance(message, str):
            return []
        words = simple_word_tokenize(message)
        return [word for word in words if word and word not in STOPWORDS]
    
    # Load the CSV file
    try:
        # First, try to inspect the file to see if it has headers
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Check if the first line looks like a header or data
            has_headers = False
            if 'author' in first_line.lower() or 'content' in first_line.lower() or 'date' in first_line.lower():
                has_headers = True
                print("CSV file appears to have headers")
            else:
                print("CSV file appears to be headerless, adding column names")
        
        if has_headers:
            # Attempt to read the CSV file, assuming it has headers
            df = pd.read_csv(csv_file, low_memory=False)
        else:
            # No headers detected, add our own
            df = pd.read_csv(csv_file, 
                            header=None, 
                            names=['AuthorID', 'Author', 'timestamp', 'content', 'Attachments', 'Reactions'],
                            low_memory=False)
            
        # Display the column names to understand the structure
        print(f"Columns in the CSV: {df.columns.tolist()}")
        
        # Try to identify key columns (case insensitive)
        column_lower = {col.lower(): col for col in df.columns}
        
        if 'timestamp' not in df.columns:
            if 'date' in column_lower:
                df.rename(columns={column_lower['date']: 'timestamp'}, inplace=True)
            elif 'Date' in df.columns:
                df.rename(columns={'Date': 'timestamp'}, inplace=True)
        
        if 'content' not in df.columns:
            if 'message' in column_lower:
                df.rename(columns={column_lower['message']: 'content'}, inplace=True)
            elif 'Content' in df.columns:
                df.rename(columns={'Content': 'content'}, inplace=True)
            
        if 'author' not in df.columns:
            if 'user' in column_lower:
                df.rename(columns={column_lower['user']: 'author'}, inplace=True)
            elif 'Author' in df.columns:
                df.rename(columns={'Author': 'author'}, inplace=True)
        
        # Check if necessary columns exist and handle column mapping
        # First create a case-insensitive mapping of existing columns
        column_mapping = {}
        column_lower_to_actual = {col.lower(): col for col in df.columns}
        
        # Define standard mappings (original name → standardized name)
        standard_mappings = {
            'date': 'timestamp',
            'time': 'timestamp',
            'datetime': 'timestamp',
            'message': 'content', 
            'text': 'content',
            'user': 'author',
            'username': 'author',
            'name': 'author'
        }
        
        # Try to map columns
        for std_col, mapped_col in [('timestamp', 'timestamp'), ('content', 'content'), ('author', 'author')]:
            # If column already exists with correct name, continue
            if std_col in df.columns:
                continue
                
            # Try exact match with alternative names
            for alt_name, std_name in standard_mappings.items():
                if std_name == std_col and alt_name in df.columns:
                    column_mapping[alt_name] = std_col
                    break
            
            # Try case-insensitive match
            if std_col not in column_mapping.values():
                for alt_name, std_name in standard_mappings.items():
                    if std_name == std_col and alt_name in column_lower_to_actual:
                        actual_col = column_lower_to_actual[alt_name]
                        column_mapping[actual_col] = std_col
                        break
            
            # Try direct capitalized version
            capitalized = std_col.capitalize()
            if capitalized in df.columns:
                column_mapping[capitalized] = std_col
        
        # Apply the mapping
        if column_mapping:
            print(f"Mapping columns: {column_mapping}")
            df.rename(columns=column_mapping, inplace=True)
        
        # Special case for Date and Content columns (your specific case)
        if 'timestamp' not in df.columns and 'Date' in df.columns:
            df.rename(columns={'Date': 'timestamp'}, inplace=True)
            
        if 'content' not in df.columns and 'Content' in df.columns:
            df.rename(columns={'Content': 'content'}, inplace=True)
            
        if 'author' not in df.columns and 'Author' in df.columns:
            df.rename(columns={'Author': 'author'}, inplace=True)
            
        # Check for missing columns again
        required_columns = ['timestamp', 'content', 'author']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Still missing required columns: {missing_columns}")
            print("Current columns:", df.columns.tolist())
            print("\nAttempting to handle mixed data types...")
            # Try reading with low_memory=False to handle mixed types
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Try one more time with explicit mappings for your file format
            if 'timestamp' not in df.columns and 'Date' in df.columns:
                df.rename(columns={'Date': 'timestamp'}, inplace=True)
                
            if 'content' not in df.columns and 'Content' in df.columns:
                df.rename(columns={'Content': 'content'}, inplace=True)
                
            if 'author' not in df.columns and 'Author' in df.columns:
                df.rename(columns={'Author': 'author'}, inplace=True)
                
            # Final check
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Failed to map required columns: {missing_columns}")
                print("Available columns:", df.columns.tolist())
                print("\nAs a fallback, will manually create required columns from available data")
                
                # Fallback: Create required columns from what's available
                if 'timestamp' not in df.columns and 'Date' in df.columns:
                    df['timestamp'] = df['Date']
                    
                if 'content' not in df.columns and 'Content' in df.columns:
                    df['content'] = df['Content']
                    
                if 'author' not in df.columns and 'Author' in df.columns:
                    df['author'] = df['Author']
                
                # Final verification
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Still cannot create required columns: {missing_columns}")
                    return None
                else:
                    print("Successfully created required columns using fallback method")
            
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Convert timestamp to datetime if not already
    if 'timestamp' in df.columns:
        print("Converting timestamps to datetime format...")
        if pd.api.types.is_string_dtype(df['timestamp']):
            try:
                # First try with UTC=True to handle timezone data properly
                print("Attempting datetime conversion with UTC=True...")
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
                
                # Check if conversion was successful
                if df['timestamp'].isna().sum() > len(df) * 0.5:
                    print("Many NaT values after conversion. Trying alternative formats...")
                    
                    # Try Discord's format with UTC flag
                    df['timestamp'] = pd.to_datetime(df['timestamp'], 
                                                   format='%Y-%m-%dT%H:%M:%S.%f%z', 
                                                   errors='coerce',
                                                   utc=True)
                
                # Check conversion success
                na_count = df['timestamp'].isna().sum()
                if na_count > 0:
                    print(f"Warning: {na_count} timestamps ({na_count/len(df)*100:.2f}%) could not be parsed.")
                else:
                    print("Timestamp conversion successful!")
                    
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                print("Creating a sequential timestamp index instead")
                df['timestamp'] = pd.Series(pd.date_range(start='2023-01-01', periods=len(df), freq='1min'))
        
        # Verify that timestamp is now datetime
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            print("WARNING: Timestamp conversion failed, creating a sequential index")
            df['timestamp'] = pd.Series(pd.date_range(start='2023-01-01', periods=len(df), freq='1min'))
    else:
        print("No timestamp column found - creating a sequential index")
        df['timestamp'] = pd.Series(pd.date_range(start='2023-01-01', periods=len(df), freq='1min'))
    
    # Create analysis dictionary to store results
    analysis = {}
    
    # 1. Basic Statistics
    analysis['total_messages'] = len(df)
    analysis['unique_users'] = df['author'].nunique()
    analysis['date_range'] = (df['timestamp'].min(), df['timestamp'].max())
    
    # 2. User Activity Analysis
    user_message_counts = df['author'].value_counts()
    analysis['most_active_users'] = user_message_counts.head(10).to_dict()
    analysis['least_active_users'] = user_message_counts.tail(10).to_dict()
    
    # 3. Temporal Analysis
    # Generate hourly activity analysis
    try:
        print("Generating hourly activity analysis...")
        # Make sure timestamp is properly converted to datetime
        if pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['hour'] = df['timestamp'].dt.hour
            hourly_activity = df.groupby('hour').size()
            analysis['hourly_activity'] = hourly_activity.to_dict()
        else:
            print("Warning: Cannot generate hourly activity - timestamp is not in datetime format")
            # Create dummy hourly data
            analysis['hourly_activity'] = {hour: 0 for hour in range(24)}
    except Exception as e:
        print(f"Error generating hourly activity: {e}")
        analysis['hourly_activity'] = {hour: 0 for hour in range(24)}
    
    # Messages by day of week
    try:
        print("Generating daily activity analysis...")
        if pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['day_of_week'] = df['timestamp'].dt.day_name()
            daily_activity = df.groupby('day_of_week').size()
            analysis['daily_activity'] = daily_activity.to_dict()
        else:
            print("Warning: Cannot generate daily activity - timestamp is not in datetime format")
            # Create dummy daily data
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            analysis['daily_activity'] = {day: 0 for day in days}
    except Exception as e:
        print(f"Error generating daily activity: {e}")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        analysis['daily_activity'] = {day: 0 for day in days}
    
    # Messages over time (weekly)
    try:
        print("Generating weekly activity analysis...")
        if pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['week'] = df['timestamp'].dt.to_period('W')
            weekly_activity = df.groupby('week').size()
            analysis['weekly_activity'] = {str(k): v for k, v in weekly_activity.items()}
        else:
            print("Warning: Cannot generate weekly activity - timestamp is not in datetime format")
            # No weekly data - skip this
            analysis['weekly_activity'] = {}
    except Exception as e:
        print(f"Error generating weekly activity: {e}")
        analysis['weekly_activity'] = {}
    
    # 4. Content Analysis
    # Average message length
    df['message_length'] = df['content'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    analysis['avg_message_length'] = df['message_length'].mean()
    
    # Most common words
    # Apply to all messages
    all_words = []
    for message in df['content'].dropna():
        all_words.extend(extract_words(message))
    
    word_freq = Counter(all_words)
    analysis['most_common_words'] = dict(word_freq.most_common(20))
    
    # 5. Sentiment Analysis
    # Add basic sentiment analysis using TextBlob
    def get_sentiment(text):
        if not isinstance(text, str) or text.strip() == '':
            return 0
        return TextBlob(text).sentiment.polarity
    
    df['sentiment'] = df['content'].apply(get_sentiment)
    analysis['avg_sentiment'] = df['sentiment'].mean()
    
    # Sentiment by user
    user_sentiment = df.groupby('author')['sentiment'].mean()
    analysis['user_sentiment'] = user_sentiment.to_dict()
    
    # 6. User Interaction Analysis
    # Find mentions (@username)
    def extract_mentions(message):
        if not isinstance(message, str):
            return []
        return re.findall(r'@(\w+)', message)
    
    df['mentions'] = df['content'].apply(extract_mentions)
    all_mentions = []
    for mentions in df['mentions']:
        all_mentions.extend(mentions)
    
    mention_freq = Counter(all_mentions)
    analysis['most_mentioned_users'] = dict(mention_freq.most_common(10))
    
    # 7. Emoji Analysis
    def extract_emojis(message):
        if not isinstance(message, str):
            return []
        # Simple emoji regex - can be expanded for better coverage
        emoji_pattern = re.compile(r'[\U0001F300-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]')
        return emoji_pattern.findall(message)
    
    df['emojis'] = df['content'].apply(extract_emojis)
    all_emojis = []
    for emojis in df['emojis']:
        all_emojis.extend(emojis)
    
    emoji_freq = Counter(all_emojis)
    analysis['most_used_emojis'] = dict(emoji_freq.most_common(10))
    
    # Generate visualizations and insights
    generate_visualizations(df, analysis, output_dir)
    
    return analysis

def generate_visualizations(df, analysis, output_dir):
    """
    Generate visualizations based on the analysis
    
    Parameters:
    df (DataFrame): The DataFrame containing the Discord chat data
    analysis (dict): The dictionary containing analysis results
    output_dir (str): The directory to save visualizations to
    """
    print("  ➡️ Setting up visualization environment...")
    # Set style
    sns.set(style="whitegrid")
    
    # Create a directory for visualizations if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  ✓ Created '{output_dir}' directory")
    
    # 1. User Activity
    print("  ➡️ Generating user activity visualization...")
    try:
        plt.figure(figsize=(12, 6))
        top_users = pd.Series(analysis['most_active_users']).sort_values(ascending=False)
        top_users.plot(kind='bar')
        plt.title('Most Active Users')
        plt.ylabel('Number of Messages')
        plt.xlabel('User')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/most_active_users.png')
        print(f"  ✓ Saved most_active_users.png to {output_dir}")
    except Exception as e:
        print(f"  ❌ Error generating user activity visualization: {e}")
    
    # 2. Hourly Activity
    print("  ➡️ Generating hourly activity visualization...")
    try:
        plt.figure(figsize=(12, 6))
        hourly = pd.Series(analysis['hourly_activity'])
        
        # Sort the hours numerically for proper display
        hourly = hourly.sort_index()
        
        hourly.plot(kind='bar')
        plt.title('Messages by Hour of Day')
        plt.ylabel('Number of Messages')
        plt.xlabel('Hour')
        
        # Set x-tick labels to be hours from 0-23
        plt.xticks(range(len(hourly)), hourly.index)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hourly_activity.png')
        print(f"  ✓ Saved hourly_activity.png to {output_dir}")
    except Exception as e:
        print(f"  ❌ Error generating hourly activity visualization: {e}")
    
    # 3. Daily Activity
    print("  ➡️ Generating daily activity visualization...")
    try:
        plt.figure(figsize=(12, 6))
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = pd.Series(analysis['daily_activity']).reindex(days_order)
        daily.plot(kind='bar')
        plt.title('Messages by Day of Week')
        plt.ylabel('Number of Messages')
        plt.xlabel('Day')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/daily_activity.png')
        print(f"  ✓ Saved daily_activity.png to {output_dir}")
    except Exception as e:
        print(f"  ❌ Error generating daily activity visualization: {e}")
    
    # 4. Message Length Distribution
    print("  ➡️ Generating message length distribution...")
    try:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['message_length'], bins=50, kde=True)
        plt.title('Message Length Distribution')
        plt.ylabel('Frequency')
        plt.xlabel('Message Length (characters)')
        plt.axvline(x=analysis['avg_message_length'], color='r', linestyle='--')
        plt.text(analysis['avg_message_length']*1.1, plt.ylim()[1]*0.9, 
                f'Average: {analysis["avg_message_length"]:.2f}', color='r')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/message_length.png')
        print(f"  ✓ Saved message_length.png to {output_dir}")
    except Exception as e:
        print(f"  ❌ Error generating message length visualization: {e}")
    
    # 5. Word Cloud
    print("  ➡️ Generating word cloud...")
    try:
        from wordcloud import WordCloud
        plt.figure(figsize=(12, 8))
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate_from_frequencies(analysis['most_common_words'])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/wordcloud.png')
        print(f"  ✓ Saved wordcloud.png to {output_dir}")
    except ImportError:
        print("  ⚠️ WordCloud package not found. Skipping word cloud visualization.")
        print("  ⚠️ You can install it with: pip install wordcloud")
    except Exception as e:
        print(f"  ❌ Error generating word cloud: {e}")
    
    # 6. Sentiment Analysis
    print("  ➡️ Generating sentiment distribution...")
    try:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['sentiment'], bins=30, kde=True)
        plt.title('Sentiment Distribution')
        plt.ylabel('Frequency')
        plt.xlabel('Sentiment Score')
        plt.axvline(x=analysis['avg_sentiment'], color='r', linestyle='--')
        plt.text(analysis['avg_sentiment']*1.1, plt.ylim()[1]*0.9, 
                f'Average: {analysis["avg_sentiment"]:.2f}', color='r')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment.png')
        print(f"  ✓ Saved sentiment.png to {output_dir}")
    except Exception as e:
        print(f"  ❌ Error generating sentiment visualization: {e}")
    
    # 7. User Sentiment
    print("  ➡️ Generating user sentiment comparison...")
    try:
        plt.figure(figsize=(12, 6))
        user_sentiment = pd.Series(analysis['user_sentiment']).sort_values(ascending=False)
        top_n = min(10, len(user_sentiment))
        user_sentiment.head(top_n).plot(kind='bar', color=sns.color_palette("RdYlGn", top_n))
        plt.title('User Sentiment (Top 10)')
        plt.ylabel('Average Sentiment Score')
        plt.xlabel('User')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/user_sentiment.png')
        print(f"  ✓ Saved user_sentiment.png to {output_dir}")
    except Exception as e:
        print(f"  ❌ Error generating user sentiment visualization: {e}")
    
    # 8. Activity Over Time (if weekly data is available)
    if 'weekly_activity' in analysis and analysis['weekly_activity']:
        print("  ➡️ Generating weekly activity trend...")
        try:
            plt.figure(figsize=(15, 6))
            weekly = pd.Series({pd.Period(k): v for k, v in analysis['weekly_activity'].items()})
            weekly.sort_index().plot(kind='line', marker='o')
            plt.title('Messages Over Time (Weekly)')
            plt.ylabel('Number of Messages')
            plt.xlabel('Week')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/weekly_activity.png')
            print(f"  ✓ Saved weekly_activity.png to {output_dir}")
        except Exception as e:
            print(f"  ❌ Error generating weekly activity visualization: {e}")
    else:
        print("  ⚠️ Skipping weekly activity visualization (insufficient data)")
    
    plt.close('all')
    print("  ✅ Visualizations complete")

def generate_insights(analysis):
    """
    Generate insights based on the analysis
    
    Parameters:
    analysis (dict): The dictionary containing analysis results
    
    Returns:
    list: A list of insights as strings
    """
    print("  ➡️ Generating insights from analysis data...")
    insights = []
    
    # Basic stats
    insights.append(f"Total messages analyzed: {analysis['total_messages']}")
    print("  ✓ Added basic stats insight")
    
    insights.append(f"Number of unique users: {analysis['unique_users']}")
    
    # Activity summary
    if analysis['date_range'][0] and analysis['date_range'][1]:
        date_diff = (analysis['date_range'][1] - analysis['date_range'][0]).days
        if date_diff > 0:
            avg_daily = analysis['total_messages'] / date_diff
            insights.append(f"Average of {avg_daily:.1f} messages per day over {date_diff} days")
            print("  ✓ Added activity summary insight")
    
    # Most active periods
    if analysis['hourly_activity']:
        try:
            max_hour = max(analysis['hourly_activity'], key=analysis['hourly_activity'].get)
            insights.append(f"Most active hour: {max_hour}:00 - {max_hour}:59")
            print("  ✓ Added hourly activity insight")
        except Exception as e:
            print(f"  ⚠️ Could not determine most active hour: {e}")
    
    if analysis['daily_activity']:
        try:
            max_day = max(analysis['daily_activity'], key=analysis['daily_activity'].get)
            insights.append(f"Most active day: {max_day}")
            print("  ✓ Added daily activity insight")
        except Exception as e:
            print(f"  ⚠️ Could not determine most active day: {e}")
    
    # User activity
    if analysis['most_active_users']:
        try:
            most_active_user = max(analysis['most_active_users'], key=analysis['most_active_users'].get)
            insights.append(f"Most active user: {most_active_user} ({analysis['most_active_users'][most_active_user]} messages)")
            
            # Calculate dominance of top user
            top_user_pct = (analysis['most_active_users'][most_active_user] / analysis['total_messages']) * 100
            if top_user_pct > 30:
                insights.append(f"The most active user contributes {top_user_pct:.1f}% of all messages, indicating a dominant presence")
            print("  ✓ Added user activity insights")
        except Exception as e:
            print(f"  ⚠️ Could not determine most active user: {e}")
    
    # Word usage
    if analysis['most_common_words']:
        try:
            top_word = max(analysis['most_common_words'], key=analysis['most_common_words'].get)
            insights.append(f"Most common word: '{top_word}' (used {analysis['most_common_words'][top_word]} times)")
            
            # Get top 3 words for more context
            top_words = sorted(analysis['most_common_words'].items(), key=lambda x: x[1], reverse=True)[:3]
            if len(top_words) >= 3:
                insights.append(f"Top 3 frequently used words: '{top_words[0][0]}', '{top_words[1][0]}', and '{top_words[2][0]}'")
            print("  ✓ Added word usage insights")
        except Exception as e:
            print(f"  ⚠️ Could not analyze word usage: {e}")
    
    # Sentiment
    try:
        insights.append(f"Average sentiment: {analysis['avg_sentiment']:.4f}")
        if analysis['avg_sentiment'] > 0.2:
            insights.append("The server has a generally positive atmosphere")
        elif analysis['avg_sentiment'] < -0.2:
            insights.append("The server has a generally negative atmosphere")
        else:
            insights.append("The server has a neutral emotional tone")
        print("  ✓ Added sentiment insights")
    except Exception as e:
        print(f"  ⚠️ Could not analyze sentiment: {e}")
    
    # Message length
    try:
        insights.append(f"Average message length: {analysis['avg_message_length']:.2f} characters")
        
        if analysis['avg_message_length'] < 20:
            insights.append("Messages tend to be very short, suggesting quick exchanges rather than in-depth discussions")
        elif analysis['avg_message_length'] > 100:
            insights.append("Messages tend to be longer, suggesting more detailed and thoughtful discussion")
        print("  ✓ Added message length insights")
    except Exception as e:
        print(f"  ⚠️ Could not analyze message length: {e}")
    
    # User engagement
    if analysis['most_mentioned_users']:
        try:
            most_mentioned = max(analysis['most_mentioned_users'], key=analysis['most_mentioned_users'].get)
            insights.append(f"Most mentioned user: @{most_mentioned} (mentioned {analysis['most_mentioned_users'][most_mentioned]} times)")
            print("  ✓ Added user engagement insight")
        except Exception as e:
            print(f"  ⚠️ Could not analyze mentions: {e}")
    
    # Emoji usage
    if analysis['most_used_emojis']:
        try:
            top_emoji = max(analysis['most_used_emojis'], key=analysis['most_used_emojis'].get)
            insights.append(f"Most used emoji: {top_emoji} (used {analysis['most_used_emojis'][top_emoji]} times)")
            print("  ✓ Added emoji usage insight")
        except Exception as e:
            print(f"  ⚠️ Could not analyze emoji usage: {e}")
    
    # Weekly trends (if available)
    if analysis['weekly_activity'] and len(analysis['weekly_activity']) > 1:
        try:
            weekly_data = [(k, v) for k, v in analysis['weekly_activity'].items()]
            weekly_data.sort()
            
            # Check if activity is increasing or decreasing
            first_weeks = sum(v for k, v in weekly_data[:min(3, len(weekly_data))])
            last_weeks = sum(v for k, v in weekly_data[-min(3, len(weekly_data)):])
            
            if last_weeks > first_weeks * 1.5:
                insights.append("Server activity has been increasing over time")
            elif last_weeks < first_weeks * 0.75:
                insights.append("Server activity has been decreasing over time")
            else:
                insights.append("Server activity has remained relatively stable over time")
            print("  ✓ Added weekly trend insight")
        except Exception as e:
            print(f"  ⚠️ Could not analyze weekly trends: {e}")
    
    print(f"  ✅ Generated {len(insights)} insights")
    return insights

def generate_recommendations(analysis):
    """
    Generate recommendations based on the analysis
    
    Parameters:
    analysis (dict): The dictionary containing analysis results
    
    Returns:
    list: A list of recommendations as strings
    """
    print("  ➡️ Generating recommendations based on analysis...")
    recommendations = []
    
    # Activity recommendations
    try:
        if analysis['hourly_activity']:
            max_hour = max(analysis['hourly_activity'], key=analysis['hourly_activity'].get)
            min_hour = min(analysis['hourly_activity'], key=analysis['hourly_activity'].get)
            
            if max_hour in range(9, 18):  # Business hours
                recommendations.append("Consider scheduling important announcements during peak hours "
                                    f"(around {max_hour}:00) to reach more users.")
                print("  ✓ Added peak hour recommendation")
            
            # Find active time blocks
            active_hours = sorted([h for h, c in analysis['hourly_activity'].items() 
                                if c > sum(analysis['hourly_activity'].values())/len(analysis['hourly_activity'])])
            if active_hours:
                active_blocks = []
                current_block = [active_hours[0]]
                
                for i in range(1, len(active_hours)):
                    if active_hours[i] == active_hours[i-1] + 1:
                        current_block.append(active_hours[i])
                    else:
                        if len(current_block) >= 2:
                            active_blocks.append(current_block)
                        current_block = [active_hours[i]]
                
                if len(current_block) >= 2:
                    active_blocks.append(current_block)
                
                if active_blocks:
                    longest_block = max(active_blocks, key=len)
                    if len(longest_block) >= 3:
                        start_hour = longest_block[0]
                        end_hour = longest_block[-1]
                        recommendations.append(f"The server is most active between {start_hour}:00 and {end_hour}:59. "
                                            "Consider scheduling community events during this time block.")
                        print("  ✓ Added active time block recommendation")
    except Exception as e:
        print(f"  ⚠️ Could not generate hourly activity recommendations: {e}")
    
    # Engagement recommendations
    try:
        if analysis['unique_users'] > 10:
            message_counts = list(analysis['most_active_users'].values())
            if len(message_counts) >= 5:
                top_5_pct = sum(sorted(message_counts, reverse=True)[:5]) / analysis['total_messages'] * 100
                
                if top_5_pct > 75:
                    recommendations.append("Server activity is heavily dominated by a few users. Consider creating more "
                                        "inclusive discussions to encourage participation from quiet members.")
                    recommendations.append("Try using polls, questions, or themed discussion days to engage less active members.")
                    print("  ✓ Added user engagement recommendations")
    except Exception as e:
        print(f"  ⚠️ Could not generate engagement recommendations: {e}")
    
    # Content recommendations
    try:
        if analysis['avg_message_length'] < 20:
            recommendations.append("Messages are generally very short. Consider introducing topics that encourage "
                                "more in-depth discussions or adding channels for specific detailed topics.")
            print("  ✓ Added message length recommendation")
    except Exception as e:
        print(f"  ⚠️ Could not generate content recommendations: {e}")
    
    # Sentiment recommendations
    try:
        if analysis['avg_sentiment'] < -0.1:
            recommendations.append("The server has a somewhat negative tone. Consider introducing positive "
                                "topics or activities to improve the atmosphere.")
            recommendations.append("Regular fun events or appreciation threads could help create a more positive environment.")
            print("  ✓ Added sentiment recommendations")
    except Exception as e:
        print(f"  ⚠️ Could not generate sentiment recommendations: {e}")
    
    # Activity balance recommendations
    try:
        if analysis['daily_activity']:
            max_day = max(analysis['daily_activity'], key=analysis['daily_activity'].get)
            min_day = min(analysis['daily_activity'], key=analysis['daily_activity'].get)
            
            if analysis['daily_activity'][max_day] > 2 * analysis['daily_activity'][min_day]:
                recommendations.append(f"Activity is much lower on {min_day}s. Consider scheduling events or "
                                    f"discussions on {min_day}s to balance weekly activity.")
                print("  ✓ Added day balance recommendation")
    except Exception as e:
        print(f"  ⚠️ Could not generate day balance recommendations: {e}")
    
    # User retention recommendations
    try:
        if 'weekly_activity' in analysis and len(analysis['weekly_activity']) > 2:
            weekly_data = [(k, v) for k, v in analysis['weekly_activity'].items()]
            weekly_data.sort()
            
            # Check if activity is decreasing in the most recent weeks
            if len(weekly_data) >= 4:
                recent_trend = weekly_data[-3:]
                if recent_trend[2][1] < recent_trend[0][1] * 0.8:
                    recommendations.append("Server activity appears to be declining recently. Consider introducing new topics, "
                                        "events, or features to reengage the community.")
                    print("  ✓ Added retention recommendation")
    except Exception as e:
        print(f"  ⚠️ Could not generate retention recommendations: {e}")
    
    # Channel recommendations based on word usage
    try:
        if analysis['most_common_words'] and len(analysis['most_common_words']) >= 10:
            topic_groups = []
            words = list(analysis['most_common_words'].keys())
            
            # Very basic topic detection
            if any(w in words for w in ['game', 'play', 'player', 'gaming']):
                topic_groups.append('gaming')
            if any(w in words for w in ['code', 'programming', 'developer', 'software']):
                topic_groups.append('programming')
            if any(w in words for w in ['music', 'song', 'band', 'listen']):
                topic_groups.append('music')
            
            if topic_groups:
                recommendations.append(f"Based on common discussion topics ({', '.join(topic_groups)}), "
                                    f"consider creating dedicated channels for these interests.")
                print("  ✓ Added channel recommendation")
    except Exception as e:
        print(f"  ⚠️ Could not generate channel recommendations: {e}")
    
    # General growth recommendation
    if analysis['unique_users'] < 50:
        recommendations.append("The server is relatively small. To grow the community, consider organizing regular "
                            "events and actively inviting new members with shared interests.")
        print("  ✓ Added growth recommendation")
    
    # If very few recommendations
    if len(recommendations) < 2:
        recommendations.append("Consider adding specific channels for different topics to organize discussions better.")
        recommendations.append("Periodic community events can help keep members engaged and increase activity.")
        print("  ✓ Added default recommendations")
    
    print(f"  ✅ Generated {len(recommendations)} recommendations")
    return recommendations

def main(csv_file):
    """
    Main function to analyze Discord chat data and print insights and recommendations
    
    Parameters:
    csv_file (str): Path to the CSV file containing Discord chat data
    """
    # Get the base filename without extension for the output directory
    import os
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = f"discord_analysis_results_{base_filename}"
    
    print(f"------------------------------------------------------------")
    print(f"Starting Discord chat analysis for: {csv_file}")
    print(f"Output will be saved to: {output_dir}")
    print(f"------------------------------------------------------------")
    
    print("\n[1/5] Analyzing chat data...")
    analysis = analyze_discord_chats(csv_file, output_dir)
    
    if analysis is None:
        print("\n❌ Analysis failed. Please check the issues reported above.")
        return
    
    print("\n[2/5] Generating insights...")
    insights = generate_insights(analysis)
    
    print("\n[3/5] Generating recommendations...")
    recommendations = generate_recommendations(analysis)
    
    print("\n[4/5] Generating visualizations...")
    # Note: visualizations are already generated in the analyze_discord_chats function
    
    # Print insights
    print("\n[5/5] Analysis complete!")
    print("\n------------------------------------------------------------")
    print("✅ INSIGHTS")
    print("------------------------------------------------------------")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Print recommendations
    print("\n------------------------------------------------------------")
    print("✅ RECOMMENDATIONS")
    print("------------------------------------------------------------")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")
    
    print("\n------------------------------------------------------------")
    print(f"Visualization results saved in the '{output_dir}' directory")
    print("------------------------------------------------------------")
    
    # Return the analysis for potential further use
    return {
        'analysis': analysis,
        'insights': insights,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python discord_analyzer.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    main(csv_file)