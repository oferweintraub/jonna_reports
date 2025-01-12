# data_extractor.py
from dotenv import load_dotenv
import os
import pymongo
from datetime import datetime, timedelta
import pandas as pd

class MongoDBExtractor:
    def __init__(self):
        load_dotenv()
        self.connection_data = {
            "uri": os.getenv("MONGO_URI"),
            "db": os.getenv("MONGO_DB"),
            "tweets_collection": os.getenv("MONGO_TWEETS_COLLECTION"),
            "users_collection": os.getenv("MONGO_USERS_COLLECTION")
        }
        self.client = None
        self.db = None
        self.tweets_collection = None
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            # Configure MongoDB client with more robust settings
            self.client = pymongo.MongoClient(
                self.connection_data["uri"],
                serverSelectionTimeoutMS=30000,  # 30 seconds timeout
                connectTimeoutMS=20000,
                socketTimeoutMS=20000,
                maxPoolSize=1,
                retryWrites=True,
                retryReads=True,
                directConnection=True,  # Force direct connection
                connect=True  # Force immediate connection attempt
            )
            
            # Test the connection explicitly
            try:
                # Force a connection attempt
                self.client.admin.command('ping')
                self.db = self.client[self.connection_data["db"]]
                self.tweets_collection = self.db[self.connection_data["tweets_collection"]]
                print("Successfully connected to MongoDB")
                
            except pymongo.errors.ServerSelectionTimeoutError as e:
                print("\nConnection Error Details:")
                print(f"Original error: {str(e)}")
                print("\nTroubleshooting steps:")
                print("1. Your IP address might need to be whitelisted")
                print("2. Check VPN settings if you're using one")
                print("3. Verify the MongoDB server status")
                print("\nConnection URI being used (without credentials):")
                uri = self.connection_data["uri"]
                safe_uri = uri.replace(uri.split('@')[0], 'mongodb://[credentials-hidden]')
                print(safe_uri)
                raise
                
            except pymongo.errors.OperationFailure as e:
                print("\nAuthentication Error:")
                print(f"Original error: {str(e)}")
                print("\nTroubleshooting steps:")
                print("1. Verify your database credentials")
                print("2. Check if your database user has the correct permissions")
                print("3. Verify the database name and collection names")
                raise
                
        except Exception as e:
            print(f"\nUnexpected error while connecting to MongoDB:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nPlease contact the system administrator with the above error details.")
            raise
        
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            
    def extract_user_tweets(self, username, start_date=None, end_date=None):
        """
        Extract tweets for a specific user with optional date range
        Args:
            username: Twitter username
            start_date: datetime object for start date
            end_date: datetime object for end date
        Returns: List of tweet documents
        """
        # Define projection to get only needed fields
        projection = {
            "author_username": 1,
            "text": 1,
            "created_at": 1,
            "retweet_count": 1,
            "like_count": 1,
            "impression_count": 1,
            "reply_count": 1,
            "_id": 0
        }
        
        # Build query
        query = {"author_username": username}
        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date
                
        # Execute query
        tweets = list(self.tweets_collection.find(query, projection))
        return tweets
    
    def extract_tweets_by_date_range(self, reference_date, days_back, usernames, period_label=None):
        """
        Extract tweets for specified users within a date range
        Args:
            reference_date: End date for tweet extraction (format: YYYY-MM-DD)
            days_back: Number of days before reference_date to extract
            usernames: List of Twitter usernames to extract tweets for
            period_label: Optional label for the period (e.g., 'pre_war', 'post_war')
        Returns:
            DataFrame containing tweets
        """
        print("Successfully connected to MongoDB")
        
        # Convert reference date to datetime
        ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
        ref_date = ref_date.replace(hour=23, minute=59, second=59)
        
        # Calculate start date
        start_date = ref_date - timedelta(days=days_back)
        start_date = start_date.replace(hour=0, minute=0, second=0)
        
        print(f"Fetching tweets from {start_date} to {ref_date}")
        
        # Convert to timestamps for MongoDB
        start_ts = int(start_date.timestamp())
        end_ts = int(ref_date.timestamp())
        
        print(f"Using timestamps from {start_ts} to {end_ts}")
        
        # Fetch tweets for each user
        all_tweets = []
        for username in usernames:
            user_tweets = self.extract_user_tweets(username, start_ts, end_ts)
            print(f"Fetched {len(user_tweets)} tweets for {username}")
            all_tweets.extend(user_tweets)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_tweets)
        
        # Save raw data
        if not df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if period_label:
                # Save in period-specific subdirectory
                save_dir = os.path.join('data', 'raw', period_label)
                os.makedirs(save_dir, exist_ok=True)
                filename = f'tweets_{period_label}_{timestamp}.csv'
            else:
                # Save in main raw directory (backward compatibility)
                save_dir = os.path.join('data', 'raw')
                os.makedirs(save_dir, exist_ok=True)
                filename = f'tweets_{start_date.strftime("%Y%m%d")}_to_{ref_date.strftime("%Y%m%d")}_{timestamp}.csv'
            
            filepath = os.path.join(save_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved raw data to: {filepath}")
            
        return df

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()