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
    
    # Display basic stats
    st.subheader("📈 Overview Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Total Songs", f"{len(filtered_df):,}")

    with stat_col2:
        avg_popularity = round(filtered_df['track_popularity'].mean(), 1)
        st.metric("Avg. Popularity", f"{avg_popularity}")

    with stat_col3:
        avg_dance = round(filtered_df['danceability'].mean(), 2)
        st.metric("Avg. Danceability", f"{avg_dance}")

    with stat_col4:
        avg_energy = round(filtered_df['energy'].mean(), 2)
        st.metric("Avg. Energy", f"{avg_energy}")

    # Row 1: Main visualizations
    st.subheader("🎵 Audio Features Analysis")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        # Radar chart for audio features
        st.write("##### Audio Feature Comparison")
        features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
        avg_features = filtered_df[features].mean().values.tolist()

        # Create radar chart
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        # Plot the average values
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        avg_features = avg_features + [avg_features[0]]  # Close the loop
        angles = angles + [angles[0]]  # Close the loop
        features = features + [features[0]]  # Close the loop

        ax.plot(angles, avg_features, 'o-', linewidth=2, color="#BB86FC")
        ax.fill(angles, avg_features, alpha=0.25, color="#BB86FC")

        # Set labels and style
        ax.set_thetagrids(np.degrees(angles[:-1]), features[:-1])
        ax.set_ylim(0, 1)
        ax.set_facecolor('#121212')
        ax.tick_params(colors="#BB86FC")
        plt.title('Audio Feature Profile', color="#BB86FC", y=1.1)

        st.pyplot(fig)

    with row1_col2:
        st.write("##### Genres & Their Subgenres")
        genres = {
            "Pop": ["Dance Pop", "Electropop", "Indie Poptimism"],
            "Rock": ["Classic Rock", "Hard Rock", "Permanent Wave"],
            "Latin": ["Reggaeton", "Latin Pop", "Tropical"],
            "R&B": ["Neo Soul", "Urban Contemporary", "Hip Pop"],
            "Rap": ["Trap", "Gangster Rap", "Southern Hip Hop"],
            "EDM": ["Big Room", "Electro House", "Progressive Electro House"]
        }

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        labels = list(genres.keys())
        subgenres = [", ".join(sub) for sub in genres.values()]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

        # Add bars for each genre
        for i, (label, subgenre) in enumerate(zip(labels, subgenres)):
            ax.bar(angles[i], 1, width=0.3, color=plt.cm.tab10(i), edgecolor="white", alpha=0.8)
            ax.text(angles[i], 1.2, label, ha='center', va='center', fontsize=12, color="#BB86FC")
            ax.text(angles[i], 0.5, subgenre, ha='center', va='center', fontsize=10, wrap=True, color="#BB86FC")

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title("Genres & Their Subgenres", fontsize=14, color="#BB86FC", pad=20)
        ax.set_facecolor('#121212')

        st.pyplot(fig)

    # Row 2: Feature Distributions
    st.subheader("📊 Feature Distributions")

    # Select feature for analysis
    selected_feature = st.selectbox(
        "Select Feature",
        ["danceability", "energy", "valence", "acousticness", "instrumentalness", "tempo", "loudness", "speechiness"]
    )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # Distribution comparison by genre
        st.write(f"##### {selected_feature.capitalize()} by Genre")
        
        # Create boxplot for feature across genres
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='playlist_genre', y=selected_feature, data=filtered_df, palette="plasma", ax=ax)
        
        ax.set_facecolor('#121212')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f"{selected_feature.capitalize()} Distribution by Genre", color="#BB86FC")
        ax.set_xlabel("Genre", color="#BB86FC")
        ax.set_ylabel(selected_feature.capitalize(), color="#BB86FC")
        ax.tick_params(colors="#BB86FC")
        plt.setp(ax.spines.values(), color="#BB86FC")
        
        plt.tight_layout()
        st.pyplot(fig)

    with row2_col2:
        # Scatter plot comparing features
        st.write("##### Feature Relationships")
        
        # Select second feature for comparison
        other_features = [f for f in ["danceability", "energy", "valence", "acousticness", "tempo"] if f != selected_feature]
        compare_feature = st.selectbox("Compare With", other_features)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=selected_feature, y=compare_feature, data=filtered_df, hue='playlist_genre', palette="plasma", alpha=0.7, ax=ax)
        
        ax.set_facecolor('#121212')
        ax.set_title(f"{selected_feature.capitalize()} vs {compare_feature.capitalize()}", color="#BB86FC")
        ax.set_xlabel(selected_feature.capitalize(), color="#BB86FC")
        ax.set_ylabel(compare_feature.capitalize(), color="#BB86FC")
        ax.tick_params(colors="#BB86FC")
        plt.setp(ax.spines.values(), color="#BB86FC")
        
        # Get legend to display nicely
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Genre")
        plt.tight_layout()
        st.pyplot(fig)

    # Row 3: Temporal analysis
    st.subheader("⏳ Trends Over Time")
    filtered_df['year'] = filtered_df['track_album_release_date'].str[:4].astype(int)

    # Group data by year and calculate means
    yearly_data = filtered_df.groupby('year').agg({
        'track_popularity': 'mean',
        'danceability': 'mean',
        'energy': 'mean',
        'valence': 'mean',
        'track_id': 'count'  # Count songs per year
    }).reset_index()
    yearly_data.rename(columns={'track_id': 'song_count'}, inplace=True)

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        # Line chart for features over time
        st.write("##### Audio Features Trends")
        
        # Allow user to select which features to display
        time_features = st.multiselect(
            "Select Features to Display",
            ['danceability', 'energy', 'valence', 'track_popularity'],
            default=['danceability', 'energy', 'valence']
        )
        
        if time_features:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for col in time_features:
                label = "Popularity" if col == "track_popularity" else col.capitalize()
                value_col = col
                
                # Normalize popularity to 0-1 scale if selected
                if col == 'track_popularity':
                    yearly_data['track_popularity_norm'] = yearly_data['track_popularity'] / 100
                    value_col = 'track_popularity_norm'
                    
                ax.plot(yearly_data['year'], yearly_data[value_col], 
                        label=label, linewidth=2, marker='o')
            
            ax.set_facecolor('#121212')
            ax.set_title("Audio Features Over Time", color="#BB86FC")
            ax.set_xlabel("Year", color="#BB86FC")
            ax.set_ylabel("Feature Value (0-1 scale)", color="#BB86FC")
            ax.tick_params(colors="#BB86FC")
            ax.legend(facecolor='#121212', edgecolor='#BB86FC', labelcolor='#BB86FC')
            plt.setp(ax.spines.values(), color="#BB86FC")
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Select at least one feature to display")

    with row3_col2:
        # Song count by year
        st.write("##### Music Production by Year")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='year', y='song_count', data=yearly_data, color="#BB86FC", ax=ax)
        
        ax.set_facecolor('#121212')
        ax.set_title("Number of Songs Released by Year", color="#BB86FC")
        ax.set_xlabel("Year", color="#BB86FC")
        ax.set_ylabel("Number of Songs", color="#BB86FC")
        ax.tick_params(colors="#BB86FC", rotation=45)
        plt.setp(ax.spines.values(), color="#BB86FC")
        
        plt.tight_layout()
        st.pyplot(fig)

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