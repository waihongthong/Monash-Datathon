import pandas as pd
import numpy as np
import re
import emoji
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# For NLP and ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy

# For handling large datasets efficiently
import dask.dataframe as dd
from dask.distributed import Client

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class YouTubeDataProcessor:
    """Main class for processing YouTube comments and videos data"""
    
    def __init__(self, comments_path=None, videos_path=None, use_dask=True):
        self.use_dask = use_dask
        self.comments_path = comments_path
        self.videos_path = videos_path
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize beauty-related keywords for categorization
        self.beauty_categories = {
            'skincare': ['skincare', 'moisturizer', 'cleanser', 'serum', 'toner', 'sunscreen', 'acne', 'wrinkle', 'anti-aging'],
            'makeup': ['makeup', 'foundation', 'lipstick', 'eyeshadow', 'mascara', 'concealer', 'blush', 'contour'],
            'fragrance': ['perfume', 'fragrance', 'cologne', 'scent', 'eau de toilette', 'eau de parfum'],
            'haircare': ['shampoo', 'conditioner', 'hair mask', 'styling', 'hair oil', 'treatment']
        }
        
        # Trend detection parameters
        self.trend_threshold = 0.3  # 30% increase in engagement
        self.viral_view_threshold = 100000  # 100k views minimum for trend consideration
        
    def load_data(self):
        """Load data using Dask for large datasets or pandas for smaller ones"""
        if self.use_dask:
            print("Loading data with Dask for large dataset processing...")
            self.comments_df = dd.read_csv(self.comments_path, dtype={'videoId': 'str', 'authorId': 'str'})
            self.videos_df = dd.read_csv(self.videos_path, dtype={'videoId': 'str', 'channelId': 'str'})
        else:
            print("Loading data with Pandas...")
            self.comments_df = pd.read_csv(self.comments_path, dtype={'videoId': 'str', 'authorId': 'str'})
            self.videos_df = pd.read_csv(self.videos_path, dtype={'videoId': 'str', 'channelId': 'str'})
        
        print(f"Comments dataset shape: {self.comments_df.shape if hasattr(self.comments_df, 'shape') else 'Dask DataFrame'}")
        print(f"Videos dataset shape: {self.videos_df.shape if hasattr(self.videos_df, 'shape') else 'Dask DataFrame'}")
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Handle emoji encoding issues
        try:
            text = emoji.demojize(text, delimiters=(" ", " "))
        except:
            pass
        
        # Clean special characters but keep hashtags and mentions
        text = re.sub(r'[^\w\s#@]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_hashtags_and_keywords(self, text):
        """Extract hashtags and relevant keywords from text"""
        if pd.isna(text):
            return [], []
        
        text = str(text)
        hashtags = re.findall(r'#\w+', text.lower())
        mentions = re.findall(r'@\w+', text.lower())
        
        # Extract beauty-related keywords
        keywords = []
        text_lower = text.lower()
        for category, terms in self.beauty_categories.items():
            for term in terms:
                if term in text_lower:
                    keywords.append(term)
        
        return hashtags, keywords
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER sentiment analyzer"""
        if pd.isna(text) or text == "":
            return 0, 'neutral'
        
        scores = self.sia.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return compound, sentiment
    
    def detect_spam(self, text, like_count=0):
        """Simple spam detection based on patterns"""
        if pd.isna(text):
            return True
        
        text = str(text).lower()
        spam_indicators = [
            len(text) < 5,  # Too short
            text.count('!') > 3,  # Too many exclamation marks
            text.count('www.') > 0 or text.count('http') > 0,  # Contains links
            len(set(text.split())) / len(text.split()) < 0.5 if len(text.split()) > 0 else True,  # Too repetitive
            bool(re.search(r'(.)\1{4,}', text)),  # Repeated characters
        ]
        
        spam_score = sum(spam_indicators) / len(spam_indicators)
        return spam_score > 0.4  # Threshold for spam classification
    
    def categorize_comment(self, text):
        """Categorize comments into beauty categories"""
        if pd.isna(text):
            return 'other'
        
        text = str(text).lower()
        category_scores = {}
        
        for category, keywords in self.beauty_categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        if not any(category_scores.values()):
            return 'other'
        
        return max(category_scores, key=category_scores.get)
    
    def calculate_engagement_rate(self, video_data):
        """Calculate engagement rate for videos"""
        # Avoid division by zero
        view_count = max(video_data.get('viewCount', 1), 1)
        like_count = video_data.get('likeCount', 0)
        comment_count = video_data.get('commentCount', 0)
        
        engagement_rate = (like_count + comment_count) / view_count
        return engagement_rate
    
    def process_comments_batch(self, batch_size=10000):
        """Process comments in batches to handle large datasets"""
        print("Processing comments for sentiment and categorization...")
        
        if self.use_dask:
            # For Dask processing
            comments_processed = self.comments_df.map_partitions(
                lambda df: self._process_comment_partition(df), 
                meta=pd.DataFrame(columns=['commentId', 'sentiment_score', 'sentiment_label', 
                                         'category', 'is_spam', 'hashtags', 'keywords'])
            )
            return comments_processed.compute()
        else:
            # For Pandas processing
            results = []
            total_rows = len(self.comments_df)
            
            for i in range(0, total_rows, batch_size):
                batch = self.comments_df.iloc[i:i+batch_size].copy()
                processed_batch = self._process_comment_partition(batch)
                results.append(processed_batch)
                
                if i % (batch_size * 10) == 0:
                    print(f"Processed {i:,} / {total_rows:,} comments")
            
            return pd.concat(results, ignore_index=True)
    
    def _process_comment_partition(self, df):
        """Process a partition/batch of comments"""
        results = []
        
        for idx, row in df.iterrows():
            text = self.preprocess_text(row.get('textOriginal', ''))
            
            # Sentiment analysis
            sentiment_score, sentiment_label = self.analyze_sentiment(text)
            
            # Category classification
            category = self.categorize_comment(text)
            
            # Spam detection
            is_spam = self.detect_spam(text, row.get('likeCount', 0))
            
            # Extract hashtags and keywords
            hashtags, keywords = self.extract_hashtags_and_keywords(row.get('textOriginal', ''))
            
            results.append({
                'commentId': row.get('commentId', ''),
                'videoId': row.get('videoId', ''),
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'category': category,
                'is_spam': is_spam,
                'hashtags': ','.join(hashtags),
                'keywords': ','.join(keywords),
                'like_count': row.get('likeCount', 0),
                'published_at': row.get('publishedAt', '')
            })
        
        return pd.DataFrame(results)

class TrendSpotter:
    """Trend detection and analysis system"""
    
    def __init__(self, processor):
        self.processor = processor
        self.trend_window_days = 7  # Look for trends in 7-day windows
        
    def detect_emerging_trends(self, comments_df, videos_df):
        """Detect emerging trends based on engagement patterns"""
        print("Detecting emerging trends...")
        
        # Convert to pandas if using Dask
        if hasattr(comments_df, 'compute'):
            comments_df = comments_df.compute()
        if hasattr(videos_df, 'compute'):
            videos_df = videos_df.compute()
        
        # Parse dates
        comments_df['published_date'] = pd.to_datetime(comments_df['published_at'], errors='coerce')
        videos_df['published_date'] = pd.to_datetime(videos_df['publishedAt'], errors='coerce')
        
        # Group by video and calculate trend metrics
        video_trends = []
        
        for video_id, video_comments in comments_df.groupby('videoId'):
            video_info = videos_df[videos_df['videoId'] == video_id]
            if video_info.empty:
                continue
            
            video_info = video_info.iloc[0]
            
            # Calculate engagement metrics over time
            engagement_over_time = self._calculate_engagement_timeline(video_comments)
            
            # Detect if this shows trending behavior
            trend_score = self._calculate_trend_score(engagement_over_time, video_info)
            
            if trend_score > self.processor.trend_threshold:
                # Extract trending keywords and hashtags
                trending_elements = self._extract_trending_elements(video_comments)
                
                video_trends.append({
                    'videoId': video_id,
                    'title': video_info.get('title', ''),
                    'trend_score': trend_score,
                    'view_count': video_info.get('viewCount', 0),
                    'like_count': video_info.get('likeCount', 0),
                    'comment_count': len(video_comments),
                    'trending_keywords': trending_elements['keywords'],
                    'trending_hashtags': trending_elements['hashtags'],
                    'dominant_sentiment': trending_elements['sentiment'],
                    'category': trending_elements['category'],
                    'published_date': video_info.get('publishedAt', ''),
                    'engagement_rate': self.processor.calculate_engagement_rate(video_info)
                })
        
        trends_df = pd.DataFrame(video_trends)
        return trends_df.sort_values('trend_score', ascending=False)
    
    def _calculate_engagement_timeline(self, comments):
        """Calculate engagement metrics over time for a video"""
        if comments.empty:
            return pd.DataFrame()
        
        # Group comments by day
        comments['date'] = comments['published_date'].dt.date
        daily_engagement = comments.groupby('date').agg({
            'like_count': 'sum',
            'commentId': 'count'
        }).rename(columns={'commentId': 'comment_count'})
        
        # Calculate rate of change
        daily_engagement['engagement_growth'] = daily_engagement['like_count'].pct_change()
        
        return daily_engagement
    
    def _calculate_trend_score(self, engagement_timeline, video_info):
        """Calculate trend score based on engagement patterns"""
        if engagement_timeline.empty:
            return 0
        
        # Factors for trend scoring
        view_count = int(video_info.get('viewCount', 0))
        recent_growth = engagement_timeline['engagement_growth'].tail(3).mean()
        
        # Normalize by view count threshold
        view_factor = min(view_count / self.processor.viral_view_threshold, 1.0)
        
        # Combine factors
        trend_score = (recent_growth if not pd.isna(recent_growth) else 0) * view_factor
        
        return max(trend_score, 0)
    
    def _extract_trending_elements(self, comments):
        """Extract trending keywords, hashtags, and patterns from comments"""
        # Combine all hashtags and keywords
        all_hashtags = []
        all_keywords = []
        sentiments = []
        categories = []
        
        for _, comment in comments.iterrows():
            if comment.get('hashtags'):
                all_hashtags.extend(comment['hashtags'].split(','))
            if comment.get('keywords'):
                all_keywords.extend(comment['keywords'].split(','))
            if comment.get('sentiment_label'):
                sentiments.append(comment['sentiment_label'])
            if comment.get('category'):
                categories.append(comment['category'])
        
        # Get top trending elements
        top_hashtags = [item[0] for item in Counter(all_hashtags).most_common(5) if item[0]]
        top_keywords = [item[0] for item in Counter(all_keywords).most_common(5) if item[0]]
        dominant_sentiment = Counter(sentiments).most_common(1)[0][0] if sentiments else 'neutral'
        dominant_category = Counter(categories).most_common(1)[0][0] if categories else 'other'
        
        return {
            'hashtags': top_hashtags,
            'keywords': top_keywords,
            'sentiment': dominant_sentiment,
            'category': dominant_category
        }
    
    def segment_audience(self, trends_df):
        """Segment audience based on engagement patterns and content preferences"""
        print("Analyzing audience segments...")
        
        segments = {
            'Gen Z Beauty Enthusiasts': trends_df[
                (trends_df['category'].isin(['makeup', 'skincare'])) & 
                (trends_df['dominant_sentiment'] == 'positive')
            ],
            'Skincare Focused': trends_df[trends_df['category'] == 'skincare'],
            'Fragrance Lovers': trends_df[trends_df['category'] == 'fragrance'],
            'High Engagement': trends_df[trends_df['engagement_rate'] > trends_df['engagement_rate'].quantile(0.8)]
        }
        
        segment_analysis = {}
        for segment_name, segment_data in segments.items():
            if not segment_data.empty:
                segment_analysis[segment_name] = {
                    'video_count': len(segment_data),
                    'avg_trend_score': segment_data['trend_score'].mean(),
                    'top_keywords': list(set([kw for sublist in segment_data['trending_keywords'] for kw in sublist]))[:10],
                    'avg_engagement_rate': segment_data['engagement_rate'].mean()
                }
        
        return segment_analysis

class CommentSense:
    """Comment quality and relevance analysis system"""
    
    def __init__(self, processor):
        self.processor = processor
        
    def analyze_comment_quality(self, comments_df):
        """Analyze comment quality and generate insights"""
        print("Analyzing comment quality and relevance...")
        
        # Convert to pandas if using Dask
        if hasattr(comments_df, 'compute'):
            comments_df = comments_df.compute()
        
        quality_metrics = {
            'total_comments': len(comments_df),
            'spam_ratio': comments_df['is_spam'].mean(),
            'quality_comment_ratio': (~comments_df['is_spam']).mean(),
            'sentiment_breakdown': comments_df['sentiment_label'].value_counts().to_dict(),
            'category_distribution': comments_df['category'].value_counts().to_dict(),
            'avg_engagement_per_comment': comments_df['like_count'].mean()
        }
        
        return quality_metrics
    
    def generate_insights_dashboard(self, comments_df, videos_df):
        """Generate insights for dashboard visualization"""
        print("Generating dashboard insights...")
        
        # Convert to pandas if using Dask
        if hasattr(comments_df, 'compute'):
            comments_df = comments_df.compute()
        if hasattr(videos_df, 'compute'):
            videos_df = videos_df.compute()
        
        # Quality comments analysis
        quality_comments = comments_df[~comments_df['is_spam']]
        
        insights = {
            'comment_quality': {
                'total_comments': len(comments_df),
                'quality_comments': len(quality_comments),
                'spam_detected': len(comments_df) - len(quality_comments),
                'quality_ratio': len(quality_comments) / len(comments_df) * 100
            },
            'sentiment_analysis': {
                'positive_ratio': (quality_comments['sentiment_label'] == 'positive').mean() * 100,
                'negative_ratio': (quality_comments['sentiment_label'] == 'negative').mean() * 100,
                'neutral_ratio': (quality_comments['sentiment_label'] == 'neutral').mean() * 100
            },
            'category_insights': quality_comments['category'].value_counts().to_dict(),
            'engagement_insights': {
                'avg_likes_per_quality_comment': quality_comments['like_count'].mean(),
                'most_liked_categories': quality_comments.groupby('category')['like_count'].mean().to_dict()
            }
        }
        
        return insights
    
    def identify_key_topics(self, comments_df, n_topics=10):
        """Identify key topics using topic modeling"""
        print("Identifying key topics from comments...")
        
        # Convert to pandas if using Dask
        if hasattr(comments_df, 'compute'):
            comments_df = comments_df.compute()
        
        # Filter quality comments
        quality_comments = comments_df[~comments_df['is_spam']]
        
        if quality_comments.empty:
            return {}
        
        # Prepare text data
        texts = quality_comments['textOriginal'].fillna('').astype(str).tolist()
        texts = [self.processor.preprocess_text(text) for text in texts if len(text) > 10]
        
        if len(texts) < 100:  # Not enough data for topic modeling
            return {'error': 'Insufficient data for topic modeling'}
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                   min_df=5, max_df=0.8, ngram_range=(1, 2))
        
        try:
            doc_term_matrix = vectorizer.fit_transform(texts[:10000])  # Limit for performance
            
            # Topic modeling
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, 
                                          max_iter=10, learning_method='batch')
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = {}
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:]]
                topics[f'Topic_{topic_idx}'] = top_words
            
            return topics
        
        except Exception as e:
            print(f"Error in topic modeling: {str(e)}")
            return {'error': f'Topic modeling failed: {str(e)}'}

def main():
    """Main execution function"""
    print("L'Oréal x Monash Datathon: TrendSpotter & CommentSense Solution")
    print("=" * 60)
    
    # Initialize processor
    # Replace these paths with your actual CSV file paths
    comments_path = "comments.csv"  # Update this path
    videos_path = "videos.csv"      # Update this path
    
    processor = YouTubeDataProcessor(comments_path, videos_path, use_dask=False)
    
    try:
        # Load data
        processor.load_data()
        
        # Process comments
        processed_comments = processor.process_comments_batch()
        
        # Initialize analysis modules
        trend_spotter = TrendSpotter(processor)
        comment_sense = CommentSense(processor)
        
        # TRENDSPOTTER ANALYSIS
        print("\n" + "="*50)
        print("TRENDSPOTTER ANALYSIS")
        print("="*50)
        
        # Detect trends
        trends = trend_spotter.detect_emerging_trends(processed_comments, processor.videos_df)
        print(f"\nDetected {len(trends)} trending videos")
        
        if not trends.empty:
            print("\nTop 5 Trending Videos:")
            for idx, trend in trends.head().iterrows():
                print(f"- {trend['title'][:50]}... (Score: {trend['trend_score']:.3f})")
        
        # Audience segmentation
        segments = trend_spotter.segment_audience(trends)
        print(f"\nIdentified {len(segments)} audience segments:")
        for segment, data in segments.items():
            print(f"- {segment}: {data['video_count']} videos, avg engagement: {data['avg_engagement_rate']:.4f}")
        
        # COMMENTSENSE ANALYSIS
        print("\n" + "="*50)
        print("COMMENTSENSE ANALYSIS")
        print("="*50)
        
        # Quality analysis
        quality_metrics = comment_sense.analyze_comment_quality(processed_comments)
        print(f"\nComment Quality Metrics:")
        print(f"- Total Comments: {quality_metrics['total_comments']:,}")
        print(f"- Quality Comment Ratio: {quality_metrics['quality_comment_ratio']:.2%}")
        print(f"- Spam Ratio: {quality_metrics['spam_ratio']:.2%}")
        
        # Dashboard insights
        dashboard_insights = comment_sense.generate_insights_dashboard(processed_comments, processor.videos_df)
        print(f"\nSentiment Breakdown:")
        for sentiment, ratio in dashboard_insights['sentiment_analysis'].items():
            print(f"- {sentiment.title()}: {ratio:.1f}%")
        
        # Topic modeling
        topics = comment_sense.identify_key_topics(processed_comments)
        if 'error' not in topics:
            print(f"\nTop Discussion Topics:")
            for topic_name, words in list(topics.items())[:3]:
                print(f"- {topic_name}: {', '.join(words[-5:])}")
        
        # Export results
        print("\n" + "="*50)
        print("EXPORTING RESULTS")
        print("="*50)
        
        # Save processed data
        processed_comments.to_csv('processed_comments.csv', index=False)
        trends.to_csv('detected_trends.csv', index=False)
        
        print("Results exported successfully!")
        print("- processed_comments.csv: All comments with quality and sentiment analysis")
        print("- detected_trends.csv: Trending videos with analysis")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please update the file paths in the main() function.")
        print(f"Current paths: {comments_path}, {videos_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()