import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to load and clean CSV data by removing duplicate lines
def load_and_clean_csv(file_path):
    data = pd.read_csv(file_path)
    data = data.drop_duplicates()
    return data

# Load and clean CSV data into a pandas DataFrame
file_path = 'pause_fluxes.csv'
data = load_and_clean_csv(file_path)
data['Cumulative Pause'] = data.groupby('Person')['Pause Before'].cumsum()

# Group data by 'Person'
grouped = data.groupby('Person')

# Define y-axis columns
y_columns = ["Size", "Left Span", "Right Span", "Weight", "RL Ratio", "WS Ratio"]

# Adjust vertical_spacing for better subplot layout
total_subplots = len(grouped)
height_per_subplot = 400
fig = make_subplots(rows=total_subplots, cols=1, shared_xaxes=False, vertical_spacing=0.001)

# Initialize sliders
sliders = [{
    'steps': [],
    'currentvalue': {
        'prefix': 'Measure: '
    },
    'pad': {"t": 50}
}]

# Calculate total traces per subplot (1 pause point + number of measures)
traces_per_subplot = 1 + len(y_columns)
annotations = []
# Loop to add traces for each measure and pause points
for row_num, (person, group) in enumerate(grouped, start=1):
    # Add pause points for each person, always visible
    fig.add_trace(
        go.Scatter(x=group['Cumulative Pause'], y=[0] * len(group), mode='markers',
                    name=f'{person} Pause Before', visible=True, showlegend=False, marker=dict(color="black"), text=group['Pause Before'], hoverinfo='text'),
        row=row_num, col=1
    )
    annotations.append(dict(
    xref='paper', x=0.89,  # Adjust `x` to position the title in the center of the subplot
    yref='paper',
    y=1 - (row_num - 0.89) / len(grouped),  # Adjust `y` to position the title above the subplot
    text=person,  # The person's name as the title
    showarrow=False,
    font=dict(size=12),  # Adjust font size as needed
    align='right'
    ))
    for measure_index, col in enumerate(y_columns):
        # Add measure-specific trace, visibility controlled by slider
        visible = (measure_index == 0)  # Only the first measure is visible initially
        fig.add_trace(
            go.Scatter(x=group['Cumulative Pause'], y=group[col], mode='lines+markers',
                       name=f'{person} {col}', visible=visible, showlegend=(row_num == 1), text = group[col], hoverinfo='text'),
            row=row_num, col=1
        )

# Corrected visibility calculation for slider steps
for measure_index, col in enumerate(y_columns):
    visibility = []
    for subplot_index in range(total_subplots):
        # Extend visibility with True for pause points and the current measure, False for others
        subplot_visibility = [False] * traces_per_subplot
        subplot_visibility[0] = True  # Pause point always visible
        subplot_visibility[measure_index + 1] = True  # Current measure visible
        visibility.extend(subplot_visibility)

    sliders[0]['steps'].append({
        'method': 'update',
        'label': col,
        'args': [{'visible': visibility},
                 {'title': f'Showing: {col}'}]
    })



# Update layout with annotations for subplot titles
fig.update_layout(annotations=annotations)
# Update layout with sliders and show figure
fig.update_layout(height=height_per_subplot * total_subplots, title_text="Flux measures after each token for each person",
                  showlegend=True, sliders=sliders)

fig.show()
fig.write_html('fluxes_graphs.html')