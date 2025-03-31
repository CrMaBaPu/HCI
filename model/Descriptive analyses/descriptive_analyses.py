import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os
from pathlib import Path
import zipfile

# ---- Helper Functions ----

def extract_metadata_from_path(file_path: str) -> tuple:
    """
    Extracts vehicle type (car/bike), criticality (critical/uncritical), and video ID from the file path.
    """
    parts = file_path.split(os.sep)
    video_folder = parts[-2]  # Example: "crit_bike_02" or "uncrit_bike_04"

    vehicle_type = "car" if "car" in video_folder else "bike" if "bike" in video_folder else "unknown"

    # **Fixed criticality detection**
    if video_folder.startswith("crit_"):
        criticality = "critical"
    elif video_folder.startswith("uncrit_"):
        criticality = "uncritical"
    else:
        criticality = "unknown"

    video_id = video_folder.split('_')[-1]  # Extracts "02" or "04"
    
    return vehicle_type, criticality, video_id

def load_and_process_data(zip_file_path: str, output_folder: Path) -> tuple:
    """
    Loads and processes gaze and YOLO data from extracted CSV files.
    Deduplicates overlapping frames and **excludes any file ending in `_features.csv`**.
    """
    if not output_folder.exists():
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        print(f"Files extracted to {output_folder}")
    else:
        print(f"Folder {output_folder} already exists. Skipping extraction.")
    
    all_files = list(output_folder.rglob('*.csv'))

    # Filter out `_features.csv` files
    all_files = [file for file in all_files if not file.name.endswith('_features.csv')]


    gaze_data_list = []
    yolo_data_list = []

    for file in all_files:
        vehicle_type, criticality, video_id = extract_metadata_from_path(str(file))

        # Process Gaze Data
        if "gaze" in file.name:
            temp_df = pd.read_csv(file)
            temp_df.rename(columns={'VideoFrame': 'Frame'}, inplace=True)  # Align with YOLO naming
            temp_df['vehicle_type'] = vehicle_type
            temp_df['criticality'] = criticality
            temp_df['video_id'] = video_id
            gaze_data_list.append(temp_df)

        # Process YOLO Data
        elif "yolo" in file.name:  
            temp_df = pd.read_csv(file)
            temp_df.rename(columns={'frame': 'Frame'}, inplace=True)  # Align with Gaze naming
            temp_df['vehicle_type'] = vehicle_type
            temp_df['criticality'] = criticality
            temp_df['video_id'] = video_id
            yolo_data_list.append(temp_df)

    # Combine all the data
    gaze_data = pd.concat(gaze_data_list, ignore_index=True)
    yolo_data = pd.concat(yolo_data_list, ignore_index=True)


    print(f"Processed gaze data shape: {gaze_data.shape}")
    print(f"Processed YOLO data shape: {yolo_data.shape}")

    # Merge YOLO data with gaze data on 'Frame' to add Arduino data
    # Calculate the average Arduino value for each frame
    average_arduino_per_frame = gaze_data.groupby('Frame')['ArduinoData1'].mean().reset_index()

    # Merge the YOLO data with the average Arduino value per frame
    yolo_data = yolo_data.merge(average_arduino_per_frame, on='Frame', how='left')

    return gaze_data, yolo_data

# ---- Load Data ----
BASE_PATH = Path("C:/Users/bayer/Documents/HCI")
ZIP_FILE_PATH = BASE_PATH / "Data.zip"
OUTPUT_FOLDER = BASE_PATH / "Data/Processed_results"

# ---- Dash App Layout ----
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([ 
    # Title
    html.H1("Descriptive Analysis of Gaze and YOLO Data", style={'text-align': 'center'}),  

    # Dropdown Filters for Vehicle Type, Criticality, and Arduino Score Range
    dbc.Row([ 
        dbc.Col([ 
            html.Label("Select Vehicle Type:"),
            dcc.Dropdown(
                id='vehicle-type-dropdown',
                options=[ 
                    {'label': 'Bike', 'value': 'bike'},
                    {'label': 'Car', 'value': 'car'},
                    {'label': 'Both', 'value': 'both'}
                ], 
                value='both', 
                multi=False
            ),
        ], width=4),
        
        dbc.Col([ 
            html.Label("Select Criticality:"),
            dcc.Dropdown(
                id='criticality-dropdown',
                options=[ 
                    {'label': 'Critical', 'value': 'critical'},
                    {'label': 'Uncritical', 'value': 'uncritical'},
                    {'label': 'Both', 'value': 'both'}
                ], 
                value='both', 
                multi=False
            ),
        ], width=4),
        
        dbc.Col([ 
            html.Label("Select Arduino Score Range:"),
            dcc.RangeSlider(
                id='arduino-score-range',
                min=0,
                max=100,
                step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                value=[0, 100]
            ),
        ], width=4),
    ], style={'margin-bottom': '0px'}),  # Reduce the bottom margin for spacing

    # Graphs Section
    html.Div([ 
        # Row 1: Arduino Score Distribution | Class Distribution
        dbc.Row([ 
            dbc.Col(dcc.Graph(id='arduino-score-distribution', style={'height': '325px'}), width=5),  
            dbc.Col(dcc.Graph(id='class-distribution', style={'height': '325px'}), width=7), 
        ], style={'margin-bottom': '0px'}), 

        # Row 2: Bounding Box Centroid Distribution | Bounding Box Box Plot
        dbc.Row([ 
            dbc.Col(dcc.Graph(id='bounding-box-centroid-distribution', style={'height': '325px'}), width=8),  
            dbc.Col(dcc.Graph(id='centroid-boxplot', style={'height': '325px'}), width=4),  
        ], style={'margin-bottom': '0px'}),  

        # Row 3: Gaze Distribution | Gaze Box Plot
        dbc.Row([ 
            dbc.Col(dcc.Graph(id='gaze-distribution', style={'height': '325px'}), width=8),  
            dbc.Col(dcc.Graph(id='gaze-boxplot', style={'height': '325px'}), width=4),  
        ], style={'margin-bottom': '0px'}),  
    ]),  
])

# ---- Dash Callbacks ----
@app.callback(
    [Output('class-distribution', 'figure'),
     Output('gaze-distribution', 'figure'),
     Output('arduino-score-distribution', 'figure'),
     Output('bounding-box-centroid-distribution', 'figure'),
     Output('gaze-boxplot', 'figure'),
     Output('centroid-boxplot', 'figure')],
    [Input('vehicle-type-dropdown', 'value'),
     Input('criticality-dropdown', 'value'),
     Input('arduino-score-range', 'value')]
)
def update_graphs(vehicle_type, criticality, arduino_score_range):
    try:
        gaze_data, yolo_data = load_and_process_data(ZIP_FILE_PATH, OUTPUT_FOLDER)
    except Exception as e:
        print(f"Error loading data: {e}")
        return dash.no_update  # or return an empty figure to handle gracefully
    # Filter data based on selected filters
    filtered_gaze = gaze_data.copy()
    filtered_yolo = yolo_data.copy()

    # Filter by vehicle type
    if vehicle_type != 'both':
        filtered_gaze = filtered_gaze[filtered_gaze['vehicle_type'] == vehicle_type]
        filtered_yolo = filtered_yolo[filtered_yolo['vehicle_type'] == vehicle_type]
    
    # Filter by criticality
    if criticality != 'both':
        filtered_gaze = filtered_gaze[filtered_gaze['criticality'] == criticality]
        filtered_yolo = filtered_yolo[filtered_yolo['criticality'] == criticality]
    
    # Filter by Arduino score range
    arduino_min, arduino_max = arduino_score_range
    filtered_gaze = filtered_gaze[(filtered_gaze['ArduinoData1'] >= arduino_min) & 
                                   (filtered_gaze['ArduinoData1'] <= arduino_max)]
    filtered_yolo = filtered_yolo[(filtered_yolo['ArduinoData1'] >= arduino_min) & 
                                   (filtered_yolo['ArduinoData1'] <= arduino_max)]
    
    # --- Plot 1: Class Distribution (Number of Detections per Class) ---
    class_counts = filtered_yolo.groupby('class').size().reset_index(name='num_detections')
    class_counts = class_counts[class_counts['num_detections'] > 2000]  # Only include classes with more than 2000 detections
    class_distribution_fig = px.bar(
        class_counts, 
        x='class', 
        y='num_detections', 
        title="Number of Detections per Class",
        labels={'class': 'Class', 'num_detections': 'Number of Detections'}
    )

    # --- Plot 2: Gaze Distribution (PixelX vs PixelY) ---
    gaze_data = filtered_gaze[['PixelX', 'PixelY']]
    gaze_distribution_fig = px.scatter(
        gaze_data, 
        x='PixelX', 
        y='PixelY', 
        title="Gaze Data Distribution"
    )

    # --- Plot 3: Arduino Score Distribution (Histogram) ---
    arduino_distribution_fig = px.histogram(
        filtered_gaze, 
        x='ArduinoData1', 
        nbins=20, 
        title="Arduino Score Distribution",
        labels={'ArduinoData1': 'Arduino Score'}
    )

    # --- Plot 4: Bounding Box Centroid Distribution (Centroid X vs Centroid Y) ---
    filtered_yolo['centroid_x'] = (filtered_yolo['x_min'] + filtered_yolo['x_max']) / 2
    filtered_yolo['centroid_y'] = (filtered_yolo['y_min'] + filtered_yolo['y_max']) / 2
    centroid_data = filtered_yolo[['centroid_x', 'centroid_y']]
    bounding_box_centroid_fig = px.scatter(
        centroid_data, 
        x='centroid_x', 
        y='centroid_y', 
        title="Bounding Box Centroid Distribution"
    )


    # --- Plot 5: Gaze Box Plot (PixelX and PixelY) ---
    gaze_boxplot_fig = px.box(
        filtered_gaze, 
        y="PixelX", 
        title="Gaze Distribution"
    )
    gaze_boxplot_fig.update_traces(boxmean=True)

    # gaze_data_aggregated = filtered_gaze.groupby('Frame').agg({'PixelX': 'mean', 'PixelY': 'mean'}).reset_index()
    # gaze_boxplot_fig = px.box(
    #     gaze_data_aggregated, 
    #     y="PixelX", 
    #     title="Gaze Distribution"
    # )
    # gaze_boxplot_fig.update_traces(boxmean=True)


    # --- Plot 6: Centroid Box Plot (Centroid X and Centroid Y) ---
    centroid_boxplot_fig = px.box(
        filtered_yolo, 
        y="centroid_x", 
        title="Centroid Distribution"
    )
    centroid_boxplot_fig.update_traces(boxmean=True)

    return class_distribution_fig, gaze_distribution_fig, arduino_distribution_fig, bounding_box_centroid_fig, gaze_boxplot_fig, centroid_boxplot_fig

# ---- Run the Dash App ----
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
