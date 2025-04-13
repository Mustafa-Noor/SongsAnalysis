import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from math import pi

# Set page config first before any Streamlit commands
st.set_page_config(page_title="Spotify Dashboard", layout="wide")

# Apply dark theme
plt.style.use('dark_background')

# Load the dataset
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv'
    return pd.read_csv(url)

data = load_data()

# Sidebar
st.sidebar.title("🎵 Music Dashboard")
page = st.sidebar.radio("Select a Page", [
    "Interactive Scatter Plots",
    "Temporal Analysis",
    "Genre Distribution Analysis",
    "Audio Features Comparison",
    "Popularity Analysis"
])

if page == "Interactive Scatter Plots":
    st.title("📈 Interactive Scatter Plots")
    st.write("Compare audio features interactively.")

    # Dropdowns for feature selection
    x_feature = st.selectbox("Select X-axis Feature", data.select_dtypes(include=np.number).columns)
    y_feature = st.selectbox("Select Y-axis Feature", data.select_dtypes(include=np.number).columns)
    genre_filter = st.multiselect("Filter by Genre", data['playlist_genre'].unique(), default=data['playlist_genre'].unique())

    filtered_data = data[data['playlist_genre'].isin(genre_filter)]

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_data, x=x_feature, y=y_feature, hue='playlist_genre', ax=ax)
    ax.set_title(f"{x_feature} vs {y_feature}")
    st.pyplot(fig)

elif page == "Temporal Analysis":
    st.title("📅 Temporal Analysis")
    st.write("Analyze trends in audio features over time.")

    # Line chart for trends
    data['year'] = pd.to_datetime(data['track_album_release_date'], errors='coerce').dt.year
    feature = st.selectbox("Select Feature for Trend Analysis", ['danceability', 'energy', 'valence', 'tempo'])
    trend_data = data.groupby('year')[feature].mean().dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    trend_data.plot(ax=ax)
    ax.set_title(f"Trend of {feature} Over Time")
    ax.set_ylabel(feature)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig)

    # Heatmap for genre popularity over time
    genre_year_data = data.groupby(['year', 'playlist_genre']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(genre_year_data, cmap="coolwarm", ax=ax)
    ax.set_title("Genre Popularity Over Time")
    st.pyplot(fig)

elif page == "Genre Distribution Analysis":
    st.title("📊 Genre Distribution Analysis")
    st.write("Explore the distribution of songs across genres.")

    # Pie chart for genre distribution
    genre_counts = data['playlist_genre'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title("Genre Distribution")
    st.pyplot(fig)

    # Stacked bar chart for subgenres
    subgenre_data = data.groupby(['playlist_genre', 'playlist_subgenre']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    subgenre_data.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Subgenre Distribution by Genre")
    ax.set_ylabel("Number of Songs")
    st.pyplot(fig)

elif page == "Audio Features Comparison":
    st.title("🎵 Audio Features Comparison")
    st.write("Compare audio features across genres.")

    # Box plots for features
    feature = st.selectbox("Select Feature for Box Plot", ['danceability', 'energy', 'valence', 'tempo'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='playlist_genre', y=feature, ax=ax)
    ax.set_title(f"{feature} by Genre")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Radar chart for average features
    avg_features = data.groupby('playlist_genre')[['danceability', 'energy', 'valence', 'tempo']].mean()
    genre = st.selectbox("Select Genre for Radar Chart", avg_features.index)
    values = avg_features.loc[genre].values.flatten().tolist()
    categories = avg_features.columns.tolist()

    values += values[:1]  # Close the radar chart
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f"Audio Features for {genre}")
    st.pyplot(fig)

elif page == "Popularity Analysis":
    st.title("🌟 Popularity Analysis")
    st.write("Analyze the popularity of songs and artists.")

    # Scatter plot for popularity vs features
    feature = st.selectbox("Select Feature for Popularity Analysis", ['danceability', 'energy', 'valence', 'tempo'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=feature, y='track_popularity', hue='playlist_genre', ax=ax)
    ax.set_title(f"Popularity vs {feature}")
    st.pyplot(fig)

    # Bar chart for most popular artists
    top_artists = data.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_artists.plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Most Popular Artists")
    ax.set_ylabel("Average Popularity")
    st.pyplot(fig)
    
elif page == "Mood-Based Song Recommender":
    st.title("🧠 Mood-Based Song Recommender")
    st.write("Adjust the sliders to find songs that match your mood or try one of the mood presets.")

    # Sliders for valence, danceability, and energy
    valence = st.slider("Valence (Mood Positiveness)", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)

    # Filter songs based on sliders
    recommended_songs = data[
        (data['valence'] >= valence - 0.1) & (data['valence'] <= valence + 0.1) &
        (data['danceability'] >= danceability - 0.1) & (data['danceability'] <= danceability + 0.1) &
        (data['energy'] >= energy - 0.1) & (data['energy'] <= energy + 0.1)
    ].head(5)

    st.subheader("Top 5 Recommended Songs")
    st.table(recommended_songs[['track_name', 'track_artist', 'valence', 'danceability', 'energy']])

    # Mood presets
    st.subheader("Try Mood Presets")
    preset = st.selectbox("Select a Mood Preset", ["Happy", "Sad", "Energetic", "Relaxed"])
    if preset == "Happy":
        preset_valence, preset_danceability, preset_energy = 0.8, 0.7, 0.7
    elif preset == "Sad":
        preset_valence, preset_danceability, preset_energy = 0.2, 0.4, 0.3
    elif preset == "Energetic":
        preset_valence, preset_danceability, preset_energy = 0.6, 0.8, 0.9
    elif preset == "Relaxed":
        preset_valence, preset_danceability, preset_energy = 0.5, 0.5, 0.4

    preset_songs = data[
        (data['valence'] >= preset_valence - 0.1) & (data['valence'] <= preset_valence + 0.1) &
        (data['danceability'] >= preset_danceability - 0.1) & (data['danceability'] <= preset_danceability + 0.1) &
        (data['energy'] >= preset_energy - 0.1) & (data['energy'] <= preset_energy + 0.1)
    ].head(5)

    st.subheader("Top 5 Songs for Selected Mood")
    st.table(preset_songs[['track_name', 'track_artist', 'valence', 'danceability', 'energy']])