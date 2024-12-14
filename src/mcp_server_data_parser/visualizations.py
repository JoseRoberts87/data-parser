import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import json

def create_bar_chart_data(data: pd.Series) -> Dict:
    """Create data for a simple bar chart."""
    return {
        "type": "bar",
        "data": {
            "labels": data.index.tolist(),
            "datasets": [{
                "data": data.values.tolist(),
                "backgroundColor": "rgba(54, 162, 235, 0.5)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 1
            }]
        }
    }

def create_line_chart_data(x_data: pd.Series, y_data: pd.Series, label: str) -> Dict:
    """Create data for a line chart."""
    return {
        "type": "line",
        "data": {
            "labels": x_data.tolist(),
            "datasets": [{
                "label": label,
                "data": y_data.tolist(),
                "borderColor": "rgba(75, 192, 192, 1)",
                "tension": 0.1
            }]
        }
    }

def create_scatter_plot_data(x_data: pd.Series, y_data: pd.Series, label: str) -> Dict:
    """Create data for a scatter plot."""
    return {
        "type": "scatter",
        "data": {
            "datasets": [{
                "label": label,
                "data": [{"x": x, "y": y} for x, y in zip(x_data, y_data)],
                "backgroundColor": "rgba(255, 99, 132, 0.5)"
            }]
        }
    }

def create_pie_chart_data(data: pd.Series) -> Dict:
    """Create data for a pie chart."""
    colors = [
        'rgba(255, 99, 132, 0.5)',
        'rgba(54, 162, 235, 0.5)',
        'rgba(255, 206, 86, 0.5)',
        'rgba(75, 192, 192, 0.5)',
        'rgba(153, 102, 255, 0.5)'
    ]
    
    return {
        "type": "pie",
        "data": {
            "labels": data.index.tolist(),
            "datasets": [{
                "data": data.values.tolist(),
                "backgroundColor": colors[:len(data)],
                "borderColor": "rgba(255, 255, 255, 1)",
                "borderWidth": 1
            }]
        }
    }

def create_heatmap_data(data: pd.DataFrame) -> Dict:
    """Create data for a heatmap."""
    return {
        "type": "heatmap",
        "data": {
            "labels": data.index.tolist(),
            "datasets": [{
                "data": [
                    {"x": i, "y": j, "v": float(v)}
                    for i, row in enumerate(data.values)
                    for j, v in enumerate(row)
                ],
                "xLabels": data.columns.tolist(),
                "yLabels": data.index.tolist()
            }]
        }
    }

def create_box_plot_data(data: pd.DataFrame) -> Dict:
    """Create data for a box plot."""
    result = []
    for column in data.columns:
        column_data = data[column].dropna()
        q1 = column_data.quantile(0.25)
        median = column_data.quantile(0.5)
        q3 = column_data.quantile(0.75)
        iqr = q3 - q1
        whisker_low = column_data[column_data >= q1 - 1.5 * iqr].min()
        whisker_high = column_data[column_data <= q3 + 1.5 * iqr].max()
        
        result.append({
            "label": column,
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "whisker_low": float(whisker_low),
            "whisker_high": float(whisker_high),
            "outliers": column_data[
                (column_data < q1 - 1.5 * iqr) | 
                (column_data > q3 + 1.5 * iqr)
            ].tolist()
        })
    
    return {
        "type": "boxplot",
        "data": {
            "datasets": result
        }
    }

def create_histogram_data(data: pd.Series, bins: int = 20) -> Dict:
    """Create data for a histogram."""
    hist, bin_edges = np.histogram(data.dropna(), bins=bins)
    return {
        "type": "histogram",
        "data": {
            "labels": [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)],
            "datasets": [{
                "data": hist.tolist(),
                "backgroundColor": "rgba(54, 162, 235, 0.5)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 1
            }]
        }
    }

def visualize_data(df: pd.DataFrame, vis_type: str, columns: Optional[List[str]] = None, 
                   group_by: Optional[str] = None, **kwargs) -> Dict:
    """Main function to create visualization data based on type."""
    if columns and not all(col in df.columns for col in columns):
        raise ValueError("Some specified columns not found in dataset")
        
    if group_by and group_by not in df.columns:
        raise ValueError(f"Group by column '{group_by}' not found in dataset")

    if columns:
        df = df[columns]

    if vis_type == "bar":
        if group_by:
            data = df.groupby(group_by).mean()
        else:
            data = df.mean()
        return create_bar_chart_data(data)

    elif vis_type == "line":
        if not columns or len(columns) != 2:
            raise ValueError("Line chart requires exactly two columns (x and y)")
        return create_line_chart_data(df[columns[0]], df[columns[1]], columns[1])

    elif vis_type == "scatter":
        if not columns or len(columns) != 2:
            raise ValueError("Scatter plot requires exactly two columns (x and y)")
        return create_scatter_plot_data(df[columns[0]], df[columns[1]], 
                                      f"{columns[0]} vs {columns[1]}")

    elif vis_type == "pie":
        if not columns or len(columns) != 1:
            raise ValueError("Pie chart requires exactly one column")
        if group_by:
            data = df.groupby(group_by)[columns[0]].sum()
        else:
            data = df[columns[0]].value_counts()
        return create_pie_chart_data(data)

    elif vis_type == "heatmap":
        if group_by:
            data = df.pivot_table(index=group_by, columns=columns[0], 
                                values=columns[1], aggfunc='mean')
        else:
            data = df[columns].corr()
        return create_heatmap_data(data)

    elif vis_type == "boxplot":
        return create_box_plot_data(df[columns] if columns else df.select_dtypes(include=[np.number]))

    elif vis_type == "histogram":
        if not columns or len(columns) != 1:
            raise ValueError("Histogram requires exactly one column")
        bins = kwargs.get('bins', 20)
        return create_histogram_data(df[columns[0]], bins)

    else:
        raise ValueError(f"Unsupported visualization type: {vis_type}")
