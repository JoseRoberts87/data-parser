import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, List, Optional, Tuple, Union

def create_visualization(df: pd.DataFrame, vis_type: str, columns: Optional[List[str]] = None, 
                        group_by: Optional[str] = None, options: Dict = None) -> str:
    """Create visualization and return base64 encoded image."""
    if options is None:
        options = {}
    
    # Set style
    # plt.style.use('seaborn')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create visualization based on type
    if vis_type == "bar":
        create_bar_plot(df, columns, group_by, options)
    elif vis_type == "line":
        create_line_plot(df, columns, group_by, options)
    elif vis_type == "scatter":
        create_scatter_plot(df, columns, options)
    elif vis_type == "pie":
        create_pie_plot(df, columns, group_by, options)
    elif vis_type == "heatmap":
        create_heatmap_plot(df, columns, group_by, options)
    elif vis_type == "boxplot":
        create_box_plot(df, columns, group_by, options)
    elif vis_type == "histogram":
        create_histogram_plot(df, columns, options)
    else:
        raise ValueError(f"Unsupported visualization type: {vis_type}")
    
    # Add title if provided
    if "title" in options:
        plt.title(options["title"])
        
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.savefig('./out/bar.png')
    plt.close()
    
    # Encode
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

def create_bar_plot(df: pd.DataFrame, columns: List[str], group_by: Optional[str], options: Dict):
    if columns:
        df = df[columns]
    if group_by:
        plot_data = df.groupby(group_by).mean()
        plot_data = plot_data.sort_values(by='price' ,ascending=False).head(10)
        plot_data.plot(kind='bar')
        plt.xticks(rotation=45)
        # raise ValueError("Bar plot with group by not supported")
    else:
        if len(columns) == 1:
            sns.barplot(data=df, y=columns[0])
        else:
            df[columns].mean().plot(kind='bar')
            plt.xticks(rotation=45)
    plt.xlabel(group_by if group_by else "Variables")
    plt.ylabel("Value")
    # plt.savefig('./out/bar.png')

def create_line_plot(df: pd.DataFrame, columns: List[str], group_by: Optional[str], options: Dict):
    if len(columns) < 2:
        raise ValueError("Line plot requires at least two columns")
    
    if group_by:
        grouped = df.groupby(group_by)[columns[1]].mean()
        plt.plot(grouped.index, grouped.values)
    else:
        plt.plot(df[columns[0]], df[columns[1]])
    
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])

def create_scatter_plot(df: pd.DataFrame, columns: List[str], options: Dict):
    if len(columns) != 2:
        raise ValueError("Scatter plot requires exactly two columns")
    
    plt.scatter(df[columns[0]], df[columns[1]], alpha=0.5)
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])

def create_pie_plot(df: pd.DataFrame, columns: List[str], group_by: Optional[str], options: Dict):
    if len(columns) != 1:
        raise ValueError("Pie chart requires exactly one column")
    
    if group_by:
        data = df.groupby(group_by)[columns[0]].sum()
    else:
        data = df[columns[0]].value_counts()
    
    plt.pie(data, labels=data.index, autopct='%1.1f%%')

def create_heatmap_plot(df: pd.DataFrame, columns: List[str], group_by: Optional[str], options: Dict):
    if group_by:
        pivot_table = df.pivot_table(
            index=group_by,
            columns=columns[0],
            values=columns[1],
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
    else:
        correlation_matrix = df[columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')

def create_box_plot(df: pd.DataFrame, columns: List[str], group_by: Optional[str], options: Dict):
    if group_by:
        sns.boxplot(data=df, x=group_by, y=columns[0])
        plt.xticks(rotation=45)
    else:
        sns.boxplot(data=df[columns])
        plt.xticks(rotation=45)

def create_histogram_plot(df: pd.DataFrame, columns: List[str], options: Dict):
    if len(columns) != 1:
        raise ValueError("Histogram requires exactly one column")
    
    bins = options.get('bins', 30)
    sns.histplot(data=df, x=columns[0], bins=bins)
    plt.xlabel(columns[0])
    plt.ylabel("Count")