import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import your main processing classes
from main import YouTubeDataProcessor, TrendSpotter, CommentSense

# Page configuration
st.set_page_config(
    page_title = "L'Oréal TrendSpotter & CommentSense Dashboard",
    page_icon = "💄",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main - header {
        font- size: 2.5rem;
color: #FF6B6B;
text - align: center;
margin - bottom: 2rem;
    }
    .metric - card {
    background: linear - gradient(135deg, #667eea 0 %, #764ba2 100 %);
    padding: 1rem;
    border - radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
    .trend - card {
    background: linear - gradient(135deg, #f093fb 0 %, #f5576c 100 %);
    padding: 1rem;
    border - radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
    .sidebar - content {
    background - color: #f8f9fa;
    padding: 1rem;
    border - radius: 10px;
}
</style >
    """, unsafe_allow_html=True)

@st.cache_data
def load_processed_data():
"""Load processed data with caching"""
try:
processed_comments = pd.read_csv('processed_comments.csv')
detected_trends = pd.read_csv('detected_trends.csv')
return processed_comments, detected_trends, True
    except FileNotFoundError:
return None, None, False

@st.cache_data
def load_raw_data():
"""Load raw data for processing"""
try:
comments_df = pd.read_csv('comments.csv')
videos_df = pd.read_csv('videos.csv')
return comments_df, videos_df, True
    except FileNotFoundError:
return None, None, False

def create_sentiment_pie_chart(comments_df):
"""Create sentiment distribution pie chart"""
sentiment_counts = comments_df['sentiment_label'].value_counts()

fig = px.pie(
    values = sentiment_counts.values,
    names = sentiment_counts.index,
    title = "Comment Sentiment Distribution",
    color_discrete_map = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107'
    }
)
fig.update_layout(height = 400)
return fig

def create_category_bar_chart(comments_df):
"""Create category distribution bar chart"""
category_counts = comments_df['category'].value_counts()

fig = px.bar(
    x = category_counts.index,
    y = category_counts.values,
    title = "Comments by Beauty Category",
    color = category_counts.values,
    color_continuous_scale = "viridis"
)
fig.update_layout(
    xaxis_title = "Category",
    yaxis_title = "Number of Comments",
    height = 400
)
return fig

def create_trend_scatter_plot(trends_df):
"""Create trend analysis scatter plot"""
fig = px.scatter(
    trends_df,
    x = 'view_count',
    y = 'trend_score',
    size = 'comment_count',
    color = 'category',
    hover_data = ['title', 'engagement_rate'],
    title = "Trend Analysis: Views vs Trend Score",
    log_x = True
)
fig.update_layout(
    xaxis_title = "View Count (log scale)",
    yaxis_title = "Trend Score",
    height = 500
)
return fig

def create_engagement_timeline(comments_df):
"""Create engagement over time timeline"""
if 'published_at' not in comments_df.columns:
return None

comments_df['published_date'] = pd.to_datetime(comments_df['published_at'], errors = 'coerce')
daily_engagement = comments_df.groupby(comments_df['published_date'].dt.date).agg({
    'like_count': 'sum',
    'commentId': 'count'
}).rename(columns = { 'commentId': 'comment_count' })

fig = go.Figure()
fig.add_trace(go.Scatter(
    x = daily_engagement.index,
    y = daily_engagement['like_count'],
    mode = 'lines+markers',
    name = 'Total Likes',
    yaxis = 'y1'
))
fig.add_trace(go.Scatter(
    x = daily_engagement.index,
    y = daily_engagement['comment_count'],
    mode = 'lines+markers',
    name = 'Comment Count',
    yaxis = 'y2',
    line = dict(color = 'orange')
))

fig.update_layout(
    title = "Engagement Timeline",
    xaxis_title = "Date",
    yaxis = dict(title = "Total Likes", side = 'left'),
    yaxis2 = dict(title = "Comment Count", side = 'right', overlaying = 'y'),
    height = 400
)
return fig

def create_quality_metrics_chart(quality_metrics):
"""Create quality metrics visualization"""
metrics = [
    ('Quality Comments', quality_metrics['quality_comment_ratio'] * 100),
    ('Spam Comments', quality_metrics['spam_ratio'] * 100)
]

fig = go.Figure(data = [
    go.Bar(
        x = [m[0] for m in metrics],
y = [m[1] for m in metrics],
marker_color = ['#28a745', '#dc3545']
        )
    ])

fig.update_layout(
    title = "Comment Quality Metrics (%)",
    yaxis_title = "Percentage",
    height = 400
)
return fig

def create_top_keywords_chart(trends_df):
"""Create top keywords visualization"""
all_keywords = []
for keywords_str in trends_df['trending_keywords'].dropna():
    if isinstance(keywords_str, str):
            # Handle string representation of lists
keywords = keywords_str.strip("[]").replace("'", "").split(", ")
all_keywords.extend([kw.strip() for kw in keywords if kw.strip()])

if all_keywords:
    keyword_counts = Counter(all_keywords)
top_keywords = dict(keyword_counts.most_common(10))

fig = px.bar(
    x = list(top_keywords.values()),
    y = list(top_keywords.keys()),
    orientation = 'h',
    title = "Top Trending Keywords",
    color = list(top_keywords.values()),
    color_continuous_scale = "plasma"
)
fig.update_layout(
    xaxis_title = "Frequency",
    yaxis_title = "Keywords",
    height = 500
)
return fig
return None

def main():
    # Header
st.markdown('<h1 class="main-header">L\'Oréal TrendSpotter & CommentSense Dashboard</h1>', unsafe_allow_html = True)
    
    # Sidebar
with st.sidebar:
st.header("Navigation")
page = st.selectbox(
    "Choose Analysis View",
    ["Overview", "TrendSpotter Analysis", "CommentSense Analysis", "Data Processing"]
)

st.markdown("---")
st.subheader("Data Status")
        
        # Check data availability
processed_comments, detected_trends, processed_available = load_processed_data()
raw_comments, raw_videos, raw_available = load_raw_data()

if processed_available:
    st.success("✅ Processed data available")
st.info(f"Comments: {len(processed_comments):,}")
st.info(f"Trends: {len(detected_trends):,}")
        elif raw_available:
st.warning("⚠️ Only raw data available")
st.info(f"Raw comments: {len(raw_comments):,}")
st.info(f"Raw videos: {len(raw_videos):,}")
        else:
st.error("❌ No data files found")
st.markdown("Please ensure the following files are in the working directory:")
st.code("- comments.csv\n- videos.csv\n- processed_comments.csv\n- detected_trends.csv")
    
    # Main content based on selected page
if page == "Overview":
    show_overview_page(processed_comments, detected_trends, processed_available)
    elif page == "TrendSpotter Analysis":
show_trendspotter_page(detected_trends, processed_available)
    elif page == "CommentSense Analysis":
show_commentsense_page(processed_comments, processed_available)
    elif page == "Data Processing":
show_processing_page(raw_comments, raw_videos, raw_available)

def show_overview_page(processed_comments, detected_trends, processed_available):
if not processed_available:
    st.error("Processed data not available. Please run the main processing script first or use the Data Processing page.")
return

st.header("Executive Summary")
    
    # Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
total_comments = len(processed_comments)
st.metric("Total Comments", f"{total_comments:,}")

with col2:
quality_ratio = (~processed_comments['is_spam']).mean()
st.metric("Quality Comments", f"{quality_ratio:.1%}")

with col3:
total_trends = len(detected_trends) if detected_trends is not None else 0
st.metric("Trending Videos", f"{total_trends:,}")

with col4:
avg_engagement = processed_comments['like_count'].mean()
st.metric("Avg Likes/Comment", f"{avg_engagement:.1f}")
    
    # Charts
col1, col2 = st.columns(2)

with col1:
sentiment_chart = create_sentiment_pie_chart(processed_comments)
st.plotly_chart(sentiment_chart, use_container_width = True)

with col2:
category_chart = create_category_bar_chart(processed_comments)
st.plotly_chart(category_chart, use_container_width = True)
    
    # Engagement timeline
if 'published_at' in processed_comments.columns:
    timeline_chart = create_engagement_timeline(processed_comments)
if timeline_chart:
    st.plotly_chart(timeline_chart, use_container_width = True)

def show_trendspotter_page(detected_trends, processed_available):
st.header("🔥 TrendSpotter Analysis")

if not processed_available or detected_trends is None or detected_trends.empty:
st.error("No trend data available. Please run the analysis first.")
return
    
    # Trend metrics
col1, col2, col3 = st.columns(3)

with col1:
st.metric("Total Trending Videos", len(detected_trends))

with col2:
avg_trend_score = detected_trends['trend_score'].mean()
st.metric("Average Trend Score", f"{avg_trend_score:.3f}")

with col3:
top_category = detected_trends['category'].mode()[0] if len(detected_trends['category'].mode()) > 0 else "N/A"
st.metric("Top Trending Category", top_category.title())
    
    # Trend analysis scatter plot
if len(detected_trends) > 0:
    trend_scatter = create_trend_scatter_plot(detected_trends)
st.plotly_chart(trend_scatter, use_container_width = True)
    
    # Top trending videos table
st.subheader("Top Trending Videos")
display_trends = detected_trends.head(10)[['title', 'trend_score', 'view_count', 'category', 'dominant_sentiment']]
st.dataframe(display_trends, use_container_width = True)
    
    # Top keywords
keywords_chart = create_top_keywords_chart(detected_trends)
if keywords_chart:
    st.plotly_chart(keywords_chart, use_container_width = True)
    
    # Audience segments
st.subheader("Audience Segments Analysis")
if 'category' in detected_trends.columns:
    segment_data = detected_trends.groupby('category').agg({
        'trend_score': 'mean',
        'view_count': 'mean',
        'engagement_rate': 'mean'
    }).round(3)
st.dataframe(segment_data, use_container_width = True)

def show_commentsense_page(processed_comments, processed_available):
st.header("💬 CommentSense Analysis")

if not processed_available:
    st.error("Processed comments data not available.")
return
    
    # Quality metrics
quality_metrics = {
    'total_comments': len(processed_comments),
    'spam_ratio': processed_comments['is_spam'].mean(),
    'quality_comment_ratio': (~processed_comments['is_spam']).mean(),
    'sentiment_breakdown': processed_comments['sentiment_label'].value_counts().to_dict(),
    'category_distribution': processed_comments['category'].value_counts().to_dict()
}
    
    # Display quality metrics
col1, col2 = st.columns(2)

with col1:
st.subheader("Comment Quality Overview")
st.metric("Total Comments", f"{quality_metrics['total_comments']:,}")
st.metric("Quality Ratio", f"{quality_metrics['quality_comment_ratio']:.1%}")
st.metric("Spam Ratio", f"{quality_metrics['spam_ratio']:.1%}")

with col2:
quality_chart = create_quality_metrics_chart(quality_metrics)
st.plotly_chart(quality_chart, use_container_width = True)
    
    # Sentiment and category analysis
col1, col2 = st.columns(2)

with col1:
sentiment_chart = create_sentiment_pie_chart(processed_comments)
st.plotly_chart(sentiment_chart, use_container_width = True)

with col2:
category_chart = create_category_bar_chart(processed_comments)
st.plotly_chart(category_chart, use_container_width = True)
    
    # Detailed analysis
st.subheader("Category-wise Sentiment Analysis")
category_sentiment = pd.crosstab(processed_comments['category'], processed_comments['sentiment_label'])
st.dataframe(category_sentiment, use_container_width = True)
    
    # Top performing comments
st.subheader("Top Performing Comments by Likes")
quality_comments = processed_comments[~processed_comments['is_spam']]
top_comments = quality_comments.nlargest(10, 'like_count')[['videoId', 'category', 'sentiment_label', 'like_count']]
st.dataframe(top_comments, use_container_width = True)

def show_processing_page(raw_comments, raw_videos, raw_available):
st.header("📊 Data Processing")

if not raw_available:
    st.error("Raw data files not found. Please ensure comments.csv and videos.csv are in the working directory.")
return

st.success("Raw data files detected!")
    
    # Data overview
col1, col2 = st.columns(2)

with col1:
st.subheader("Comments Dataset")
st.info(f"Shape: {raw_comments.shape}")
st.dataframe(raw_comments.head(), use_container_width = True)

with col2:
st.subheader("Videos Dataset")
st.info(f"Shape: {raw_videos.shape}")
st.dataframe(raw_videos.head(), use_container_width = True)
    
    # Processing button
st.subheader("Run Analysis")

if st.button("🚀 Run TrendSpotter & CommentSense Analysis", type = "primary"):
    progress_bar = st.progress(0)
status_text = st.empty()

try:
            # Initialize processor
status_text.text("Initializing processor...")
processor = YouTubeDataProcessor(use_dask = False)
processor.comments_df = raw_comments
processor.videos_df = raw_videos
progress_bar.progress(20)
            
            # Process comments
status_text.text("Processing comments...")
processed_comments = processor.process_comments_batch()
progress_bar.progress(60)
            
            # Run trend analysis
status_text.text("Analyzing trends...")
trend_spotter = TrendSpotter(processor)
trends = trend_spotter.detect_emerging_trends(processed_comments, raw_videos)
progress_bar.progress(80)
            
            # Save results
status_text.text("Saving results...")
processed_comments.to_csv('processed_comments.csv', index = False)
trends.to_csv('detected_trends.csv', index = False)
progress_bar.progress(100)

status_text.text("✅ Analysis complete!")
st.success("Analysis completed successfully! Results saved to CSV files.")
st.balloons()
            
            # Show quick results
st.subheader("Quick Results")
col1, col2, col3 = st.columns(3)

with col1:
st.metric("Processed Comments", len(processed_comments))
with col2:
st.metric("Detected Trends", len(trends))
with col3:
quality_ratio = (~processed_comments['is_spam']).mean()
st.metric("Quality Ratio", f"{quality_ratio:.1%}")
            
        except Exception as e:
st.error(f"An error occurred during processing: {str(e)}")
status_text.text("❌ Processing failed")

st.markdown("---")
st.info("💡 **Tip**: After running the analysis, navigate to other pages to explore the results in detail!")

if __name__ == "__main__":
    main()