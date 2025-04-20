import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set up the page
st.set_page_config(layout="wide", page_title="Muzify", page_icon="🎧")

# Custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
    }
    
    .main {
        background-color: #000000; /* Black background */
        color: #BB86FC; /* App's primary text color */
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

df = pd.read_csv("spotify_songs.csv")

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
st.title("MUZIFY")
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
    st.subheader("🎵 Audio Features Dashboard")
  
    # Row 1: Future Trend & Genre Wheel Donut
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("### <span style='color:#BB86FC'>Future Trend</span>", unsafe_allow_html=True)
        filtered_df['release_year'] = pd.to_datetime(filtered_df['track_album_release_date'], errors='coerce').dt.year
        yearly_popularity = filtered_df.groupby('release_year')['track_popularity'].mean().dropna()

        fig, ax = plt.subplots()
        ax.plot(yearly_popularity.index, yearly_popularity.values, marker='o', color="#BB86FC", linewidth=2)
        ax.set_facecolor('#121212')
        ax.set_title("Average Track Popularity by Year", color="#BB86FC")
        ax.set_xlabel("Release Year", color="#BB86FC")
        ax.set_ylabel("Popularity", color="#BB86FC")
        ax.tick_params(colors="#BB86FC")
        st.pyplot(fig)

    with row1_col2:
        st.markdown("### <span style='color:#BB86FC'>Interactive Genre Wheel</span>", unsafe_allow_html=True)
        genre_counts = filtered_df['playlist_genre'].value_counts()
        labels = genre_counts.index.tolist()
        sizes = genre_counts.values.tolist()
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts = ax.pie(sizes, labels=labels, startangle=90, colors=colors, textprops={'color':'#BB86FC'})
        centre_circle = plt.Circle((0, 0), 0.60, fc='#121212')
        fig.gca().add_artist(centre_circle)
        ax.set_facecolor('#121212')
        ax.set_title("Genre Distribution", color="#BB86FC")
        st.pyplot(fig)

    # Row 2: Mood Matrix & Guess-the-Song Interface
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.write("##### Mood Matrix (Valence vs Energy)")
        fig, ax = plt.subplots()
        ax.set_facecolor('#121212')
        scatter = ax.scatter(
            filtered_df['valence'],
            filtered_df['energy'],
            alpha=0.6,
            c=filtered_df['danceability'],
            cmap='cool',
            edgecolors='w'
        )
        ax.set_xlabel('Valence', color="#BB86FC")
        ax.set_ylabel('Energy', color="#BB86FC")
        ax.set_title("Mood Matrix", color="#BB86FC")
        ax.tick_params(colors="#BB86FC")
        plt.colorbar(scatter, label='Danceability')
        st.pyplot(fig)

    with row2_col2:
        st.write("##### Guess The Song Mood")
        random_song = filtered_df.sample(1)
        features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
        song_features = random_song[features].values.flatten().tolist()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        song_features += [song_features[0]]
        angles += [angles[0]]
        features += [features[0]]

        ax.plot(angles, song_features, 'o-', linewidth=2, color="#03DAC6")
        ax.fill(angles, song_features, alpha=0.25, color="#03DAC6")
        ax.set_thetagrids(np.degrees(angles[:-1]), features[:-1])
        ax.set_ylim(0, 1)
        ax.set_facecolor('#121212')
        ax.tick_params(colors="#03DAC6")
        plt.title('Can You Guess the Mood?', color="#03DAC6", y=1.1)

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
    st.markdown("<h2 style='color:#BB86FC;'>⏳ <b>Trends Over Time</b></h2>", unsafe_allow_html=True)
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
        st.markdown("##### 🎚️ <span style='color:#BB86FC'>Audio Features Trends</span>", unsafe_allow_html=True)
    
        time_features = st.multiselect(
            "Select Features to Display",
            ['danceability', 'energy', 'valence', 'track_popularity'],
            default=['danceability', 'energy', 'valence']
        )
    
        if time_features:
            with st.spinner("Generating interactive trend chart..."):
                plot_df = yearly_data.copy()
                plot_df['track_popularity_norm'] = plot_df['track_popularity'] / 100
    
                # Melt data to long format for plotly
                melted = pd.melt(plot_df, id_vars='year', 
                                 value_vars=[f if f != 'track_popularity' else 'track_popularity_norm' for f in time_features],
                                 var_name='Feature', value_name='Value')
                
                # Rename 'track_popularity_norm' back to 'Popularity' for display
                melted['Feature'] = melted['Feature'].replace({'track_popularity_norm': 'Popularity'})
                melted['Feature'] = melted['Feature'].str.capitalize()
    
                fig = px.line(
                    melted, x='year', y='Value', color='Feature',
                    markers=True, template='plotly_dark',
                    title="Audio Features Over Time"
                )
                fig.update_traces(line=dict(width=3))
                fig.update_layout(
                    title_font_color='#BB86FC',
                    legend_font_color='#BB86FC',
                    font=dict(color="#BB86FC"),
                    plot_bgcolor='#121212',
                    paper_bgcolor='#121212',
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one feature to display")
    
    with row3_col2:
        st.markdown("##### 📊 <span style='color:#BB86FC'>Music Production by Year</span>", unsafe_allow_html=True)
    
        with st.spinner("Loading song production stats..."):
            fig2 = px.bar(
                yearly_data, x='year', y='song_count',
                labels={'song_count': 'Number of Songs', 'year': 'Year'},
                template='plotly_dark',
                color_discrete_sequence=['#BB86FC']
            )
            fig2.update_layout(
                title="Number of Songs Released by Year",
                title_font_color='#BB86FC',
                font=dict(color="#BB86FC"),
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig2, use_container_width=True)
    
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
    
    # User inputs for filtering songs
    genre_input = st.selectbox("Select Genre", df['playlist_genre'].unique())
    energy_input = st.slider("Select Energy Level", 0.0, 1.0, 0.5)
    valence_input = st.slider("Select Valence (Mood Positiveness)", 0.0, 1.0, 0.5)
    danceability_input = st.slider("Select Danceability", 0.0, 1.0, 0.5)

    # Filter songs based on user inputs
    recommended_songs = df[
        (df['playlist_genre'] == genre_input) &
        (df['energy'] >= energy_input - 0.1) & (df['energy'] <= energy_input + 0.1) &
        (df['valence'] >= valence_input - 0.1) & (df['valence'] <= valence_input + 0.1) &
        (df['danceability'] >= danceability_input - 0.1) & (df['danceability'] <= danceability_input + 0.1)
    ].head(5)

    # Display recommended songs
    st.subheader("Recommended Songs")
    if not recommended_songs.empty:
        st.table(recommended_songs[['track_name', 'track_artist', 'energy', 'valence', 'danceability']])
    else:
        st.write("No songs match your criteria. Try adjusting the filters.")