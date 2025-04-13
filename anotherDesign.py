import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the page
st.set_page_config(layout="wide", page_title="Spotify Pulse", page_icon="🎧")

# Custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
    }
    
    .main {
        background-color: #000000; /* Black background */
        color: #BB86FC; /* Flutter app's primary text color */
    }
    
    .st-bw {
        background-color: #000000 !important;
    }
    
    h1, h2, h3 {
        color: #BB86FC !important;
        text-shadow: 0 0 10px #BB86FC;
    }
    
    .sidebar .sidebar-content {
        background-color: #1F1F1F !important;
        border-right: 1px solid #BB86FC;
    }
    
    .st-bq {
        border-color: #BB86FC;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
    return pd.read_csv(url)

df = load_data()

# Sidebar filters
st.sidebar.title("FILTERS")
selected_genres = st.sidebar.multiselect(
    "Select genres",
    options=df['playlist_genre'].unique(),
    default=df['playlist_genre'].unique()
)

year_range = st.sidebar.slider(
    "Select release year range",
    min_value=int(df['track_album_release_date'].str[:4].min()),
    max_value=int(df['track_album_release_date'].str[:4].max()),
    value=(2010, 2020)
)

# Filter data
filtered_df = df[
    (df['playlist_genre'].isin(selected_genres)) & 
    (df['track_album_release_date'].str[:4].astype(int).between(year_range[0], year_range[1]))
]

# Dashboard title
st.title("SPOTIFY PULSE")
st.markdown("### Futuristic Music Analytics Dashboard")

# Tabs for Analysis, Top Artists, and Song Recommendation
tab1, tab2, tab3 = st.tabs(["Analysis", "Top Artists", "Song Recommendation"])

with tab1:
    st.header("📊 Analysis")
    
    # Row for selecting features
    st.subheader("Customize Analysis")
    selected_feature = st.selectbox("Select Feature for Analysis", ["track_popularity", "danceability", "energy", "valence", "tempo"])
    selected_genre = st.selectbox("Select Genre for Filtering", filtered_df['playlist_genre'].unique())
    
    # Filter data based on selected genre
    genre_filtered_df = filtered_df[filtered_df['playlist_genre'] == selected_genre]
    
    # Row for visualizations
    col1, col2, col3 = st.columns(3)

    with col1:
        # Bar graph for selected feature
        st.subheader(f"Distribution of {selected_feature.capitalize()}")
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.histplot(genre_filtered_df[selected_feature], kde=True, color="#BB86FC", ax=ax)
        ax.set_facecolor('#000000')
        ax.set_title(f"{selected_feature.capitalize()} Distribution", color="#BB86FC")
        ax.set_xlabel(selected_feature.capitalize(), color="#BB86FC")
        ax.set_ylabel("Frequency", color="#BB86FC")
        ax.tick_params(colors="#BB86FC")
        plt.setp(ax.spines.values(), color="#BB86FC")
        st.pyplot(fig)

    with col2:
        # Bar graph for genre counts
        st.subheader("Genre Distribution")
        genre_counts = filtered_df['playlist_genre'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="plasma", ax=ax)
        ax.set_facecolor('#000000')
        ax.set_title("Genre Distribution", color="#BB86FC")
        ax.set_xlabel("Number of Songs", color="#BB86FC")
        ax.set_ylabel("Genre", color="#BB86FC")
        ax.tick_params(colors="#BB86FC")
        plt.setp(ax.spines.values(), color="#BB86FC")
        st.pyplot(fig)

    with col3:
        # Line chart for temporal trends
        st.subheader("Temporal Trends")
        filtered_df['year'] = filtered_df['track_album_release_date'].str[:4].astype(int)
        yearly_data = filtered_df.groupby('year').agg({
            'track_popularity': 'mean',
            'danceability': 'mean',
            'energy': 'mean',
            'valence': 'mean'
        }).reset_index()
        fig, ax = plt.subplots(figsize=(4, 4))
        for col in ['danceability', 'energy', 'valence']:
            ax.plot(yearly_data['year'], yearly_data[col], label=col.capitalize(), linewidth=2)
        ax.legend(facecolor='#000000', edgecolor='#BB86FC', labelcolor='#BB86FC')
        ax.set_facecolor('#000000')
        ax.set_title("Audio Features Over Time", color="#BB86FC")
        ax.set_xlabel("Year", color="#BB86FC")
        ax.set_ylabel("Feature Value", color="#BB86FC")
        ax.tick_params(colors="#BB86FC")
        plt.setp(ax.spines.values(), color="#BB86FC")
        st.pyplot(fig)

    # with col1:
    #     # Genre distribution
    #     st.subheader("Genre Distribution")
    #     genre_counts = filtered_df['playlist_genre'].value_counts()
    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     ax.pie(genre_counts, labels=genre_counts.index, 
    #            colors=plt.cm.plasma(np.linspace(0, 1, len(genre_counts))),
    #            wedgeprops=dict(edgecolor='#BB86FC', linewidth=2))
    #     ax.set_facecolor('#000000')
    #     st.pyplot(fig)

    # with col2:
    #     # Feature correlation
    #     st.subheader("Feature Correlation")
    #     features = ['danceability', 'energy', 'loudness', 'speechiness', 
    #                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    #     corr = filtered_df[features].corr()
    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     sns.heatmap(corr, annot=False, cmap='plasma', cbar=False, ax=ax)
    #     ax.set_facecolor('#000000')
    #     st.pyplot(fig)

    # with col3:
    #     # Temporal trends
    #     st.subheader("Temporal Trends")
    #     filtered_df['year'] = filtered_df['track_album_release_date'].str[:4].astype(int)
    #     yearly_data = filtered_df.groupby('year').agg({
    #         'track_popularity': 'mean',
    #         'danceability': 'mean',
    #         'energy': 'mean',
    #         'valence': 'mean'
    #     }).reset_index()
    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     for col in ['danceability', 'energy', 'valence']:
    #         ax.plot(yearly_data['year'], yearly_data[col], label=col.capitalize(), linewidth=2)
    #     ax.legend(facecolor='#000000', edgecolor='#BB86FC', labelcolor='#BB86FC')
    #     ax.set_facecolor('#000000')
    #     st.pyplot(fig)

with tab2:
    st.header("Top Artists")
    
    # Dropdown to select factor
    factor = st.selectbox("Select Factor", ["Number of Songs", "Energy", "Popularity", "Danceability", "Valence", "Loudness"])
    
    if factor == "Number of Songs":
        st.subheader("Top Artists by Number of Songs")
        top_artists = filtered_df['track_artist'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=top_artists.values, y=top_artists.index, palette='plasma', ax=ax)
        ax.set_facecolor('#000000')
        plt.title("Top Artists by Number of Songs", color='#BB86FC')
        plt.xlabel("Number of Songs", color='#BB86FC')
        plt.ylabel("Artist", color='#BB86FC')
        plt.setp(ax.spines.values(), color='#BB86FC')
        ax.tick_params(colors='#BB86FC')
        st.pyplot(fig)
    
    elif factor == "Energy":
        st.subheader("Top Artists by Energy")
        top_artists_energy = filtered_df.groupby('track_artist')['energy'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=top_artists_energy.values, y=top_artists_energy.index, palette='plasma', ax=ax)
        ax.set_facecolor('#000000')
        plt.title("Top Artists by Energy", color='#BB86FC')
        plt.xlabel("Average Energy", color='#BB86FC')
        plt.ylabel("Artist", color='#BB86FC')
        plt.setp(ax.spines.values(), color='#BB86FC')
        ax.tick_params(colors='#BB86FC')
        st.pyplot(fig)
    
    elif factor == "Popularity":
        st.subheader("Top Artists by Popularity")
        top_artists_popularity = filtered_df.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=top_artists_popularity.values, y=top_artists_popularity.index, palette='plasma', ax=ax)
        ax.set_facecolor('#000000')
        plt.title("Top Artists by Popularity", color='#BB86FC')
        plt.xlabel("Average Popularity", color='#BB86FC')
        plt.ylabel("Artist", color='#BB86FC')
        plt.setp(ax.spines.values(), color='#BB86FC')
        ax.tick_params(colors='#BB86FC')
        st.pyplot(fig)
    
    elif factor == "Danceability":
        st.subheader("Top Artists by Danceability")
        top_artists_danceability = filtered_df.groupby('track_artist')['danceability'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=top_artists_danceability.values, y=top_artists_danceability.index, palette='plasma', ax=ax)
        ax.set_facecolor('#000000')
        plt.title("Top Artists by Danceability", color='#BB86FC')
        plt.xlabel("Average Danceability", color='#BB86FC')
        plt.ylabel("Artist", color='#BB86FC')
        plt.setp(ax.spines.values(), color='#BB86FC')
        ax.tick_params(colors='#BB86FC')
        st.pyplot(fig)
    
    elif factor == "Valence":
        st.subheader("Top Artists by Valence")
        top_artists_valence = filtered_df.groupby('track_artist')['valence'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=top_artists_valence.values, y=top_artists_valence.index, palette='plasma', ax=ax)
        ax.set_facecolor('#000000')
        plt.title("Top Artists by Valence", color='#BB86FC')
        plt.xlabel("Average Valence", color='#BB86FC')
        plt.ylabel("Artist", color='#BB86FC')
        plt.setp(ax.spines.values(), color='#BB86FC')
        ax.tick_params(colors='#BB86FC')
        st.pyplot(fig)
    
    elif factor == "Loudness":
        st.subheader("Top Artists by Loudness")
        top_artists_loudness = filtered_df.groupby('track_artist')['loudness'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=top_artists_loudness.values, y=top_artists_loudness.index, palette='plasma', ax=ax)
        ax.set_facecolor('#000000')
        plt.title("Top Artists by Loudness", color='#BB86FC')
        plt.xlabel("Average Loudness", color='#BB86FC')
        plt.ylabel("Artist", color='#BB86FC')
        plt.setp(ax.spines.values(), color='#BB86FC')
        ax.tick_params(colors='#BB86FC')
        st.pyplot(fig)

with tab3:
    st.header("🎵 Song Recommendation")
    genre_input = st.selectbox("Select Genre", df['playlist_genre'].unique())
    energy_input = st.slider("Select Energy Level", 0.0, 1.0, 0.5)
    valence_input = st.slider("Select Valence (Mood Positiveness)", 0.0, 1.0, 0.5)

    recommended_songs = df[
        (df['playlist_genre'] == genre_input) &
        (df['energy'] >= energy_input - 0.1) & (df['energy'] <= energy_input + 0.1) &
        (df['valence'] >= valence_input - 0.1) & (df['valence'] <= valence_input + 0.1)
    ].head(5)

    st.subheader("Recommended Songs")
    st.table(recommended_songs[['track_name', 'track_artist', 'energy', 'valence']])