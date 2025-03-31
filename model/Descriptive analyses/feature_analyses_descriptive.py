import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the dataset
df = pd.read_csv('model/features_dataset.csv')

# App layout
app.layout = html.Div(
    style={'overflowY': 'scroll', 'height': '100vh'},
    children=[
        html.H1("Feature Distributions Dashboard"),
        html.Div([
            html.Label("Select Feature:"),
            dcc.Dropdown(
                id='feature_selector',
                options=[
                    {'label': col, 'value': col} for col in df.columns if col not in ['person_id', 'category', 'criticality', 'file_id']
                ],
                value='num_per_cls_car',  # Default value
            ),
        ], style={'padding': '20px'}),
        html.Div([
            html.Label("Select Person ID:"),
            dcc.Dropdown(
                id='person_id_selector',
                options=[{'label': str(pid), 'value': pid} for pid in df['person_id'].unique()],
                value=[df['person_id'].unique()[0]],  # Default value (first person)
                multi=True,  # Allow multiple selections
                clearable=True
            ),
        ], style={'padding': '20px'}),
        html.Div([
            html.Label("Select Category:"),
            dcc.Dropdown(
                id='category_selector',
                options=[{'label': cat, 'value': cat} for cat in df['category'].unique()],
                value=[df['category'].unique()[0]],  # Default value (first category)
                multi=True,  # Allow multiple selections
                clearable=True
            ),
        ], style={'padding': '20px'}),
        html.Div([
            html.Label("Select Criticality:"),
            dcc.Dropdown(
                id='criticality_selector',
                options=[{'label': crit, 'value': crit} for crit in df['criticality'].unique()],
                value=[df['criticality'].unique()[0]],  # Default value (first criticality)
                multi=True,  # Allow multiple selections
                clearable=True
            ),
        ], style={'padding': '20px'}),
        html.Div([
            html.Label("Select File ID:"),
            dcc.Dropdown(
                id='file_id_selector',
                options=[{'label': fid, 'value': fid} for fid in df['file_id'].unique()],
                value=[df['file_id'].unique()[0]],  # Default value (first file_id)
                multi=True,  # Allow multiple selections
                clearable=True
            ),
        ], style={'padding': '20px'}),
        dcc.Graph(id='feature_distribution_graph'),
    ]
)

# Callback to update the graph based on selections
@app.callback(
    Output('feature_distribution_graph', 'figure'),
    [
        Input('feature_selector', 'value'),
        Input('person_id_selector', 'value'),
        Input('category_selector', 'value'),
        Input('criticality_selector', 'value'),
        Input('file_id_selector', 'value'),
    ]
)
def update_graph(selected_feature, selected_person_ids, selected_categories, selected_criticalities, selected_file_ids):
    # Filter the DataFrame based on selected values
    filtered_df = df[
        (df['person_id'].isin(selected_person_ids)) &
        (df['category'].isin(selected_categories)) &
        (df['criticality'].isin(selected_criticalities)) &
        (df['file_id'].isin(selected_file_ids))
    ]
    # Create the distribution plot for the selected feature
    fig = px.histogram(filtered_df, x=selected_feature, title=f'Distribution of {selected_feature}')
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
