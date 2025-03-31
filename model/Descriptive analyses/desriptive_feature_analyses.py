import pandas as pd  
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os
from pathlib import Path
import zipfile

# ---- Helper Functions ----

def load_and_process_data(output_folder: Path) -> pd.DataFrame:   
    all_files = list(output_folder.rglob('*.csv'))

    gaze_data_list, features_data_list = [], []

    for file in all_files:
        person_ID, category, criticality, file_ID

        try:
            temp_df = pd.read_csv(file)
            temp_df['vehicle_type'] = category
            temp_df['criticality'] = criticality
            temp_df['video_id'] = file_ID
            temp_df['person_id'] = person_ID

            if "features" in file.name:
                temp_df = pd.read_csv(file)
                temp_df.rename(columns={'frame': 'Frame'}, inplace=True)
                temp_df['vehicle_type'] = vehicle_type
                temp_df['criticality'] = criticality
                temp_df['video_id'] = video_id
                features_data_list.append(temp_df)
            elif "gaze" in file.name:
                temp_df = pd.read_csv(file)
                temp_df.rename(columns={'VideoFrame': 'Frame'}, inplace=True)
                temp_df['vehicle_type'] = vehicle_type
                temp_df['criticality'] = criticality
                temp_df['video_id'] = video_id
                gaze_data_list.append(temp_df)

        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    if not features_data_list or not gaze_data_list:
        return pd.DataFrame()  

    features_data = pd.concat(features_data_list, ignore_index=True)
    gaze_data = pd.concat(gaze_data_list, ignore_index=True)

    if 'Frame' in gaze_data.columns and 'Frame' in features_data.columns:
        features_data = features_data.merge(
            gaze_data[['Frame', 'ArduinoData1', 'vehicle_type', 'criticality', 'video_id']],
            on=['Frame', 'vehicle_type', 'criticality', 'video_id'], 
            how='left'
        )

    return features_data

# ---- Dash App Layout ----

BASE_PATH = Path("C:/Users/bayer/Documents/HCI")
ZIP_FILE_PATH = BASE_PATH / "Data.zip"
OUTPUT_FOLDER = BASE_PATH / "Data/Processed_results"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Descriptive Analysis of Features and Arduino Data", style={'text-align': 'center'}),  

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
                value='both', multi=False
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
                value='both', multi=False
            ),
        ], width=4),

        dbc.Col([
            html.Label("Select Arduino Score Range:"),
            dcc.RangeSlider(
                id='arduino-score-range',
                min=0, max=100, step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                value=[0, 100]
            ),
        ], width=4),
    ], style={'margin-bottom': '10px'}),

    html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(id='arduino-score-distribution'), width=12),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='total-objects-distribution'), width=6),
            dbc.Col(dcc.Graph(id='total-classes-distribution'), width=6),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='cluster-std-dev-distribution'), width=6),
            dbc.Col(dcc.Graph(id='central-detection-size-distribution'), width=6),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='num-per-class-car'), width=4),
            dbc.Col(dcc.Graph(id='num-per-class-bicycle'), width=4),
            dbc.Col(dcc.Graph(id='num-per-class-pedestrian'), width=4),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='average-size-car'), width=4),
            dbc.Col(dcc.Graph(id='average-size-bicycle'), width=4),
            dbc.Col(dcc.Graph(id='average-size-pedestrian'), width=4),
        ]),
    ]),
])

# ---- Dash Callbacks ----

@app.callback(
    [Output('arduino-score-distribution', 'figure'),
     Output('total-objects-distribution', 'figure'),
     Output('total-classes-distribution', 'figure'),
     Output('cluster-std-dev-distribution', 'figure'),
     Output('central-detection-size-distribution', 'figure'),
     Output('num-per-class-car', 'figure'),
     Output('num-per-class-bicycle', 'figure'),
     Output('num-per-class-pedestrian', 'figure'),
     Output('average-size-car', 'figure'),
     Output('average-size-bicycle', 'figure'),
     Output('average-size-pedestrian', 'figure')],
    [Input('vehicle-type-dropdown', 'value'),
     Input('criticality-dropdown', 'value'),
     Input('arduino-score-range', 'value')]
)
def update_graphs(vehicle_type, criticality, arduino_score_range):
    try:
        features_data = load_and_process_data(ZIP_FILE_PATH, OUTPUT_FOLDER)
        if features_data.empty:
            raise ValueError("No valid data found")
    except Exception as e:
        print(f"Error loading data: {e}")
        return [px.histogram(title="No Data Available") for _ in range(11)]

    filtered_features = features_data.copy()

    if vehicle_type != 'both':
        filtered_features = filtered_features[filtered_features['vehicle_type'] == vehicle_type]

    if criticality != 'both':
        filtered_features = filtered_features[filtered_features['criticality'] == criticality]

    arduino_min, arduino_max = arduino_score_range
    filtered_features = filtered_features[
        (filtered_features['ArduinoData1'] >= arduino_min) & 
        (filtered_features['ArduinoData1'] <= arduino_max)
    ]

    # Ensure all required columns exist before plotting
    required_columns = [
        'ArduinoData1', 'total_objects_all', 'total_classes', 
        'cluster_std_dev', 'central_detection_size', 
        'num_per_class_car', 'num_per_class_bicycle', 'num_per_class_pedestrian', 
        'average_size_car', 'average_size_bicycle', 'average_size_pedestrian'
    ]

    if not all(col in filtered_features.columns for col in required_columns):
        print("Warning: Missing columns in data")
        return [px.histogram(title="Missing Data") for _ in range(11)]

    return (
        px.histogram(filtered_features, x='ArduinoData1', title="Arduino Score Distribution"),
        px.histogram(filtered_features, x='total_objects_all', title="Total Objects Distribution"),
        px.histogram(filtered_features, x='total_classes', title="Total Classes Distribution"),
        px.histogram(filtered_features, x='cluster_std_dev', title="Cluster Standard Deviation"),
        px.histogram(filtered_features, x='central_detection_size', title="Central Detection Size"),
        px.box(filtered_features, y="num_per_class_car", title="Number of Cars per Frame"),
        px.box(filtered_features, y="num_per_class_bicycle", title="Number of Bicycles per Frame"),
        px.box(filtered_features, y="num_per_class_pedestrian", title="Number of Pedestrians per Frame"),
        px.box(filtered_features, y="average_size_car", title="Average Size of Cars"),
        px.box(filtered_features, y="average_size_bicycle", title="Average Size of Bicycles"),
        px.box(filtered_features, y="average_size_pedestrian", title="Average Size of Pedestrians"),
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
