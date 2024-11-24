import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import mapping, box
from folium.features import GeoJsonTooltip, GeoJsonPopup
import ast
import os
import requests  # Import requests for HTTP requests

# Import OpenAI client
from openai import OpenAI

# Set your OpenAI API key from Streamlit secrets
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
)

# Check if the API key was successfully loaded
if client.api_key is None or client.api_key == "":
    st.error("OpenAI API key not found. Please set it in Streamlit secrets.")

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Load review data
@st.cache_data
def load_review_data():
    url = "https://drive.google.com/file/d/1-JhrkatP2jp_Oiu2erDyBV9tjeh4nu56/view?usp=sharing"
    file_id = url.split("/d/")[1].split("/view")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    df = pd.read_csv(download_url)
    return df

with st.spinner("Loading data..."):
    review_df = load_review_data()

# Custom styles for dividers
horizontal_line_style = "border: 2px solid #ccc; margin: 20px 0;"
vertical_divider_style = """
    <style>
    .vertical-divider {
        width: 1px;
        background-color: #ccc;
        margin: 0 20px;
    }
    </style>
    <div class="vertical-divider"></div>
"""

# Divide the layout into two main columns (left and right)
col1, col2 = st.columns([1, 1], gap="medium")

# **Left Section: Folium Map**
with col1:
    # Add a catchy title to the map
    st.markdown("<h2 style='text-align: center;'>Explore Philadelphia's Culinary Landscape</h2>", unsafe_allow_html=True)
    
    map_height = 800  # Adjust the height as needed

    # Load the data from the CSV file
    with st.spinner("Loading map data..."):
        data = pd.read_csv("https://www.dropbox.com/scl/fi/3a48oyk9ntt6yumzpd3bf/yelp_business_data.csv?rlkey=ksemy3n0bie3fuad1xfnmbzff&st=wona6o93&dl=1")

    # Filter for only Pennsylvania (state 'PA') entries
    data_pa = data[data['state'] == 'PA']

    # Further filter for only restaurants in Philadelphia area
    # Define the approximate bounding box for Philadelphia (adjust as needed)
    philly_bounds = {
        "north": 40.137992,
        "south": 39.867004,
        "east": -74.955763,
        "west": -75.280303
    }
    data_philly = data_pa[
        (data_pa['latitude'] >= philly_bounds['south']) &
        (data_pa['latitude'] <= philly_bounds['north']) &
        (data_pa['longitude'] >= philly_bounds['west']) &
        (data_pa['longitude'] <= philly_bounds['east'])
    ]

    # Initialize the map centered on Philadelphia
    philadelphia_map = folium.Map(location=[39.9526, -75.1652], zoom_start=12)

    # Define grid size (e.g., 0.01 degrees per cell)
    lat_step = 0.01
    lon_step = 0.01

    # Initialize an empty list to store grid cell data
    grid_data = []

    # Generate grid cells within the bounding box
    lat_bins = np.arange(philly_bounds['south'], philly_bounds['north'], lat_step)
    lon_bins = np.arange(philly_bounds['west'], philly_bounds['east'], lon_step)

    for lat in lat_bins:
        for lon in lon_bins:
            # Filter restaurants within this grid cell
            cell_data = data_philly[
                (data_philly['latitude'] >= lat) &
                (data_philly['latitude'] < lat + lat_step) &
                (data_philly['longitude'] >= lon) &
                (data_philly['longitude'] < lon + lon_step)
            ]

            # Calculate the average rating for restaurants in this cell
            if len(cell_data) > 0:
                avg_rating = cell_data['stars'].mean()

                # Store grid cell information
                grid_data.append({
                    "lat_min": lat,
                    "lat_max": lat + lat_step,
                    "lon_min": lon,
                    "lon_max": lon + lon_step,
                    "average_rating": avg_rating
                })

    # Convert grid data to a DataFrame
    grid_df = pd.DataFrame(grid_data)

    # Sort the DataFrame by latitude (descending) and then longitude (ascending)
    grid_df_sorted = grid_df.sort_values(by=["lat_min", "lon_min"], ascending=[False, True])

    # Reset the index and add a sequential index column as 'box_number'
    grid_df_sorted.reset_index(drop=True, inplace=True)
    grid_df_sorted.insert(0, "box_number", range(1, len(grid_df_sorted) + 1))

    # Function to get color based on average rating (Teal and Maroon scheme)
    def get_color(avg_rating):
        if avg_rating >= 4.5:
            return "#008080"  # Teal
        elif avg_rating >= 4.0:
            return "#20B2AA"  # Light Sea Green
        elif avg_rating >= 3.5:
            return "#40E0D0"  # Turquoise
        elif avg_rating >= 3.0:
            return "#FF7F50"  # Coral
        elif avg_rating >= 2.5:
            return "#CD5C5C"  # Indian Red
        else:
            return "#800000"  # Maroon

    # Create GeoJson features
    features = []

    for idx, row in grid_df_sorted.iterrows():
        # Create a rectangle polygon for the grid cell
        polygon = box(row['lon_min'], row['lat_min'], row['lon_max'], row['lat_max'])

        # Create a feature with properties
        feature = {
            "type": "Feature",
            "id": idx,  # Add unique ID
            "geometry": mapping(polygon),
            "properties": {
                "box_number": row['box_number'],
                "average_rating": row['average_rating']
            }
        }
        features.append(feature)

    # Create a FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    # Now, create a GeoJson layer with a popup to capture clicks
    geojson_layer = folium.GeoJson(
        data=feature_collection,
        name='Grid Cells',
        style_function=lambda feature: {
            'fillColor': get_color(feature['properties']['average_rating']),
            'color': get_color(feature['properties']['average_rating']),
            'weight': 1,
            'fillOpacity': 0.3,  # Adjusted opacity back to 0.3
        },
        highlight_function=lambda feature: {'weight': 3, 'color': 'blue'},
        tooltip=GeoJsonTooltip(fields=['box_number', 'average_rating'], aliases=['Box Number:', 'Average Rating:']),
        popup=GeoJsonPopup(fields=['box_number'], labels=False)
    )

    geojson_layer.add_to(philadelphia_map)

    # Initialize box_number to None
    box_number = None

    # Render the updated map in the Streamlit interface
    map_data = st_folium(philadelphia_map, width=700, height=map_height)

    # Capture the clicked box number
    if map_data:
        # Check for possible keys that may contain click data
        clicked_data = None
        if 'last_active_drawing' in map_data and map_data['last_active_drawing'] is not None:
            clicked_data = map_data['last_active_drawing']
        elif 'last_clicked' in map_data and map_data['last_clicked'] is not None:
            clicked_data = map_data['last_clicked']
        elif 'last_object_clicked' in map_data and map_data['last_object_clicked'] is not None:
            clicked_data = map_data['last_object_clicked']

        if clicked_data and 'properties' in clicked_data:
            box_number = clicked_data['properties'].get('box_number', None)
            if box_number is not None:
                box_number = int(box_number)
                st.write(f"You clicked on box number: {box_number}")
    else:
        st.write("Click on a box to see the review distribution")

# **Right Section: Charts and Description**
with col2:
    # Top Section: Charts
    chart_col1, chart_col2 = st.columns(2)  # Two charts side by side

    # Plotly Bar Chart (Updated with data)
    with chart_col1:
        if box_number is not None:
            # Load master_csv_df from Google Drive
            @st.cache_data
            def load_master_csv():
                url = "https://drive.google.com/file/d/1LFmThCe1Hw66Iwo-sadCH-o0MqvJ4HCt/view?usp=sharing"
                file_id = url.split("/d/")[1].split("/view")[0]
                download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                df = pd.read_csv(download_url)
                return df

            with st.spinner("Loading box data..."):
                master_csv_df = load_master_csv()

            # Find the row corresponding to the selected box_number
            box_row = master_csv_df[master_csv_df['Name'] == f"box{box_number}.csv"]

            if not box_row.empty:
                box_link = box_row['Final Google Drive Links'].values[0]

                # Load the CSV file for the selected box
                @st.cache_data
                def load_box_csv(box_link):
                    try:
                        response = requests.get(box_link)
                        if response.status_code != 200:
                            return pd.DataFrame()
                        elif response.content.strip() == b'':
                            return pd.DataFrame()
                        else:
                            df = pd.read_csv(box_link)
                            return df
                    except pd.errors.EmptyDataError:
                        return pd.DataFrame()
                    except Exception as e:
                        return pd.DataFrame()

                with st.spinner("Loading box CSV data..."):
                    df = load_box_csv(box_link)

                if not df.empty:
                    # Parse the 'topic_scores' column and compute the average for each label
                    topic_scores_dicts = df['topic_scores'].apply(ast.literal_eval)

                    # Initialize a dictionary to store total scores and count for each label
                    label_scores = {}

                    for topic_scores in topic_scores_dicts:
                        labels = topic_scores['labels']
                        scores = topic_scores['scores']

                        for label, score in zip(labels, scores):
                            if label not in label_scores:
                                label_scores[label] = {'total_score': 0, 'count': 0}
                            label_scores[label]['total_score'] += score
                            label_scores[label]['count'] += 1

                    # Compute the average for each label
                    average_scores = {label: score_info['total_score'] / score_info['count'] for label, score_info in label_scores.items()}

                    # Prepare the data for visualization
                    sorted_scores = dict(sorted(average_scores.items(), key=lambda item: item[1], reverse=True))
                    labels_bar = list(sorted_scores.keys())
                    scores_bar = list(sorted_scores.values())

                    # Create a color scale based on scores (Teal to Maroon)
                    import plotly.express as px

                    # Define a custom color scale from teal to maroon
                    custom_colorscale = [
                        [0.0, '#008080'],  # Teal
                        [0.5, '#7F7F7F'],  # Grey
                        [1.0, '#800000']   # Maroon
                    ]

                    # Create the bar chart using Plotly
                    bar_chart = go.Figure(
                        data=[go.Bar(
                            x=scores_bar,
                            y=labels_bar,
                            orientation='h',
                            marker=dict(
                                color=scores_bar,
                                colorscale=custom_colorscale,
                                colorbar=dict(title='Average Score'),
                            ),
                            text=[f"{score:.2f}" for score in scores_bar],
                            textposition='auto',
                        )],
                        layout=go.Layout(
                            title=f'Average Scores by Metric for Box Number {box_number}',
                            xaxis=dict(title='Average Score'),
                            yaxis=dict(title='Metrics', autorange='reversed'),  # Highest score at the top
                            margin=dict(l=150),  # Adjust left margin to accommodate labels
                        )
                    )
                    st.plotly_chart(bar_chart, use_container_width=True)
                else:
                    st.write("No detailed ratings available for this box.")
            else:
                st.write("No detailed ratings available for this box.")
        else:
            st.write("Click on a box to see the bar chart")

    # Plotly Pie Chart (Updated with data)
    with chart_col2:
        if box_number is not None:
            # Filter the data for the selected box_number
            row = review_df[review_df['box_num'] == box_number]
            if not row.empty:
                positive_reviews = int(row["pos_revs"].values[0])
                negative_reviews = int(row["neg_revs"].values[0])

                # Only create the pie chart if there are reviews
                if positive_reviews > 0 or negative_reviews > 0:
                    labels_pie = ['Positive', 'Negative']
                    values_pie = [positive_reviews, negative_reviews]
                    colors_pie = ['#008080', '#800000']  # Teal and Maroon

                    # Create a Plotly pie chart
                    pie_chart = go.Figure(
                        data=[go.Pie(
                            labels=labels_pie,
                            values=values_pie,
                            marker=dict(colors=colors_pie)
                        )],
                        layout=go.Layout(title=f"Distribution of Reviews for Box Number {box_number}")
                    )
                    st.plotly_chart(pie_chart, use_container_width=True)
                else:
                    st.write("There are no reviews for this specific box.")
            else:
                st.write("There are no reviews for this specific box.")
        else:
            st.write("Click on a box to see the review distribution")

    # Horizontal Separator Line
    st.markdown(f'<div style="{horizontal_line_style}"></div>', unsafe_allow_html=True)

    # Bottom Section: Text Description
    st.write("**Analysis Description**")
    if box_number is not None:
        # Load master_csv_df from Google Drive
        @st.cache_data
        def load_master_csv():
            url = "https://drive.google.com/file/d/1LFmThCe1Hw66Iwo-sadCH-o0MqvJ4HCt/view?usp=sharing"
            file_id = url.split("/d/")[1].split("/view")[0]
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            df = pd.read_csv(download_url)
            return df

        with st.spinner("Loading analysis data..."):
            master_csv_df = load_master_csv()

        # Find the row corresponding to the selected box_number
        box_row = master_csv_df[master_csv_df['Name'] == f"box{box_number}.csv"]

        if not box_row.empty:
            box_link = box_row['Final Google Drive Links'].values[0]

            # Load the CSV file for the selected box
            @st.cache_data
            def load_box_csv(box_link):
                try:
                    response = requests.get(box_link)
                    if response.status_code != 200:
                        return pd.DataFrame()
                    elif response.content.strip() == b'':
                        return pd.DataFrame()
                    else:
                        df = pd.read_csv(box_link)
                        return df
                except pd.errors.EmptyDataError:
                    return pd.DataFrame()
                except Exception as e:
                    return pd.DataFrame()

            with st.spinner("Loading box CSV data..."):
                df = load_box_csv(box_link)

            if not df.empty:
                label_scores = {}

                for idx, row_topic in df.iterrows():
                    topic_data = ast.literal_eval(row_topic['topic_scores'])  # Convert string to dictionary
                    labels = topic_data['labels']
                    scores = topic_data['scores']
                    for label, score in zip(labels, scores):
                        if label not in label_scores:
                            label_scores[label] = {'total': 0, 'count': 0}
                        label_scores[label]['total'] += score
                        label_scores[label]['count'] += 1

                # Calculate the average scores for each label
                avg_scores = {label: data['total'] / data['count'] for label, data in label_scores.items()}

                # Sort the labels by their average scores in descending order
                sorted_labels = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

                # Extract the top 3 labels
                top_3_topics = [label for label, score in sorted_labels[:3]]

                # Generate the analysis description using OpenAI API

                prompt = f"Give a brief overview of insights on opening a restaurant in this location where the top 3 topics are {', '.join(top_3_topics)}. Please limit to a few sentences."

                try:
                    with st.spinner("Generating analysis description..."):
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            model="gpt-4",  # Replace with the model you have access to
                        )

                        analysis_description = chat_completion.choices[0].message.content.strip()

                    st.write(analysis_description)
                except Exception as e:
                    st.error(f"An error occurred while generating the analysis description: {e}")
            else:
                # Get the average rating from grid_df_sorted
                box_row_in_grid = grid_df_sorted[grid_df_sorted['box_number'] == box_number]
                if not box_row_in_grid.empty:
                    avg_rating = box_row_in_grid['average_rating'].values[0]

                    # Construct prompt based on avg_rating
                    prompt = f"Even though there are no written reviews for this location, the average rating is {avg_rating:.2f}. Based on this average rating, provide insights on opening a restaurant in this location. Explain whether it's an average, good, or bad place to open a new restaurant."

                    try:
                        with st.spinner("Generating analysis description..."):
                            chat_completion = client.chat.completions.create(
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt,
                                    }
                                ],
                                model="gpt-4",  # Replace with the model you have access to
                            )

                            analysis_description = chat_completion.choices[0].message.content.strip()

                        st.write(analysis_description)
                    except Exception as e:
                        st.error(f"An error occurred while generating the analysis description: {e}")
                else:
                    st.write("Average rating data is not available for this location.")
        else:
            st.write("No data available for this box.")
    else:
        st.write("Click on a box to see the analysis description.")
