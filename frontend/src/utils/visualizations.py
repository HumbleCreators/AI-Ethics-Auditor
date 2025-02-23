import plotly.graph_objects as go

def create_bar_chart(data, title="Bar Chart", xaxis_title="Categories", yaxis_title="Values"):
    """
    Creates a Plotly bar chart.
    
    :param data: A dictionary with keys "categories" and "values".
    :return: A Plotly Figure object.
    """
    fig = go.Figure([go.Bar(x=data["categories"], y=data["values"])])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig

def create_pie_chart(labels, values, title="Pie Chart"):
    """
    Creates a Plotly pie chart.
    
    :param labels: List of labels.
    :param values: List of values corresponding to each label.
    :return: A Plotly Figure object.
    """
    fig = go.Figure([go.Pie(labels=labels, values=values)])
    fig.update_layout(title=title)
    return fig

def create_heatmap(data, title="Heatmap", xaxis_title="X Axis", yaxis_title="Y Axis"):
    """
    Creates a Plotly heatmap.
    
    :param data: A 2D list (matrix) representing the heatmap values.
    :return: A Plotly Figure object.
    """
    fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis'))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig

def create_gauge_chart(value, min_value=0, max_value=10, title="Gauge Chart"):
    """
    Creates a Plotly gauge chart.
    
    :param value: The current value to display.
    :param min_value: Minimum value for the gauge.
    :param max_value: Maximum value for the gauge.
    :return: A Plotly Figure object.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {"text": title},
        gauge = {
            "axis": {"range": [min_value, max_value]},
            "steps": [
                {"range": [min_value, (min_value+max_value)/3], "color": "lightgreen"},
                {"range": [(min_value+max_value)/3, 2*(min_value+max_value)/3], "color": "yellow"},
                {"range": [2*(min_value+max_value)/3, max_value], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    return fig
