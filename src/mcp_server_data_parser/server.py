import asyncio
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Global storage for loaded datasets
datasets: Dict[str, pd.DataFrame] = {}
metadata: Dict[str, Dict[str, Any]] = {}  # Store metadata about datasets

SERVER_NAME: str = "mcp-server-data-parser"

server = Server(SERVER_NAME)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for data parsing and analysis."""
    return [
        types.Tool(
            name="load-csv",
            description="Load a CSV file into memory for analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file to load"
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Name to reference this dataset"
                    },
                    "options": {
                        "type": "object",
                        "description": "Optional pandas read_csv options",
                        "properties": {
                            "separator": {"type": "string"},
                            "encoding": {"type": "string"},
                            "skiprows": {"type": "integer"},
                            "date_columns": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": []
                    }
                },
                "required": ["file_path", "dataset_name"]
            },
        ),
        types.Tool(
            name="analyze-data",
            description="Analyze loaded dataset based on a question",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to analyze"
                    },
                    "question": {
                        "type": "string",
                        "description": "Question about the data to analyze"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific columns to analyze (optional)"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Column to group by for analysis (optional)"
                    }
                },
                "required": ["dataset_name", "question"]
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests."""
    if not arguments:
        raise ValueError("Arguments are required")

    if name == "load-csv":
        return await load_csv_file(arguments)
    elif name == "analyze-data":
        return await analyze_data(arguments)
    
    raise ValueError(f"Unknown tool: {name}")

async def load_csv_file(arguments: dict) -> list[types.TextContent]:
    """Load a CSV file into memory and store it in the datasets dictionary."""
    file_path = arguments.get("file_path")
    dataset_name = arguments.get("dataset_name")
    options = arguments.get("options", {})

    if not file_path or not dataset_name:
        raise ValueError("Both file_path and dataset_name are required")

    try:
        # Convert options to pandas read_csv parameters
        pandas_options = {
            "sep": options.get("separator", ","),
            "encoding": options.get("encoding", "utf-8"),
            "skiprows": options.get("skiprows", 0)
        }

        # Load the CSV file
        df = pd.read_csv(file_path, **pandas_options)
        
        # Handle date columns if specified
        date_columns = options.get("date_columns", [])
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        datasets[dataset_name] = df
        
        # Store metadata about the dataset
        metadata[dataset_name] = {
            "loaded_at": datetime.now(),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "date_columns": date_columns,
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns)
        }

        # Prepare summary of the loaded data
        summary = f"""
Successfully loaded dataset '{dataset_name}':
- Number of rows: {len(df):,}
- Number of columns: {len(df.columns)}
- Numeric columns: {', '.join(metadata[dataset_name]['numeric_columns'])}
- Categorical columns: {', '.join(metadata[dataset_name]['categorical_columns'])}
- Date columns: {', '.join(date_columns)}
- Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
        """

        return [types.TextContent(type="text", text=summary.strip())]

    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")

async def analyze_data(arguments: dict) -> list[types.TextContent]:
    """Analyze a loaded dataset based on a question."""
    dataset_name = arguments.get("dataset_name")
    question = arguments.get("question")
    columns = arguments.get("columns", [])
    group_by = arguments.get("group_by")

    if not dataset_name or not question:
        raise ValueError("Both dataset_name and question are required")

    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found. Please load it first using load-csv.")

    df = datasets[dataset_name]
    dataset_meta = metadata[dataset_name]
    
    try:
        # If specific columns are provided, validate them
        if columns:
            invalid_cols = [col for col in columns if col not in df.columns]
            if invalid_cols:
                raise ValueError(f"Invalid columns: {', '.join(invalid_cols)}")
            df = df[columns]

        # If group_by is provided, validate it
        if group_by and group_by not in df.columns:
            raise ValueError(f"Group by column '{group_by}' not found in dataset")
            
        response = analyze_question(df, question, dataset_meta, group_by)
        return [types.TextContent(type="text", text=response)]
    except Exception as e:
        raise ValueError(f"Error analyzing data: {str(e)}")

def analyze_question(df: pd.DataFrame, question: str, meta: dict, group_by: Optional[str] = None) -> str:
    """Enhanced analysis based on the question asked."""
    question = question.lower()
    
    # If grouping is requested, create a grouped DataFrame
    if group_by:
        grouped = df.groupby(group_by)
    
    # Statistical Analysis
    if any(word in question for word in ["mean", "average", "avg"]):
        if group_by:
            result = grouped[meta["numeric_columns"]].mean()
            return f"Average by {group_by}:\n{result.to_string()}"
        else:
            result = df[meta["numeric_columns"]].mean()
            return "Averages:\n" + "\n".join(f"{col}: {val:.2f}" for col, val in result.items())
            
    elif "median" in question:
        if group_by:
            result = grouped[meta["numeric_columns"]].median()
            return f"Median by {group_by}:\n{result.to_string()}"
        else:
            result = df[meta["numeric_columns"]].median()
            return "Medians:\n" + "\n".join(f"{col}: {val:.2f}" for col, val in result.items())
            
    elif any(word in question for word in ["std", "standard deviation"]):
        if group_by:
            result = grouped[meta["numeric_columns"]].std()
            return f"Standard deviation by {group_by}:\n{result.to_string()}"
        else:
            result = df[meta["numeric_columns"]].std()
            return "Standard deviations:\n" + "\n".join(f"{col}: {val:.2f}" for col, val in result.items())
    
    # Distribution Analysis
    elif "quantile" in question or "percentile" in question:
        quantiles = [0.25, 0.5, 0.75]
        if group_by:
            result = grouped[meta["numeric_columns"]].quantile(quantiles)
            return f"Quantiles by {group_by}:\n{result.to_string()}"
        else:
            result = df[meta["numeric_columns"]].quantile(quantiles)
            return f"Quantiles:\n{result.to_string()}"
            
    elif "distribution" in question:
        result = []
        for col in meta["numeric_columns"]:
            stats = df[col].describe()
            result.append(f"\nDistribution of {col}:")
            result.append(stats.to_string())
        return "\n".join(result)
    
    # Temporal Analysis
    elif any(word in question for word in ["trend", "time series", "over time"]) and meta["date_columns"]:
        results = []
        for date_col in meta["date_columns"]:
            for num_col in meta["numeric_columns"]:
                monthly_avg = df.set_index(date_col)[num_col].resample('M').mean()
                results.append(f"\nMonthly average of {num_col}:")
                results.append(monthly_avg.to_string())
        return "\n".join(results) if results else "No temporal analysis possible with current data"
    
    # Correlation Analysis
    elif "correlation" in question or "correlate" in question:
        corr_matrix = df[meta["numeric_columns"]].corr()
        return f"Correlation matrix:\n{corr_matrix.to_string()}"
    
    # Category Analysis
    elif "category" in question or "categorical" in question:
        results = []
        for col in meta["categorical_columns"]:
            value_counts = df[col].value_counts()
            results.append(f"\nValue counts for {col}:")
            results.append(value_counts.to_string())
        return "\n".join(results) if results else "No categorical columns found"
    
    # Missing Value Analysis
    elif "missing" in question or "null" in question:
        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df)) * 100
        result = pd.DataFrame({
            'Missing Count': missing,
            'Missing Percentage': missing_pct
        })
        return "Missing value analysis:\n" + result.to_string()
    
    # Basic Summary Statistics
    elif "summary" in question or "describe" in question:
        description = df.describe(include='all').to_string()
        return f"Comprehensive summary statistics:\n{description}"
    
    # Outlier Detection
    elif "outlier" in question:
        results = []
        for col in meta["numeric_columns"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
            results.append(f"\nOutliers in {col}:")
            results.append(f"Count: {len(outliers)}")
            if len(outliers) > 0:
                results.append(f"Values: {outliers.to_string()}")
        return "\n".join(results)
    
    # Size and Shape
    elif "count" in question or "size" in question:
        if group_by:
            result = grouped.size()
            return f"Counts by {group_by}:\n{result.to_string()}"
        else:
            return f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns"
    
    # Unique Values
    elif "unique" in question:
        results = []
        for col in df.columns:
            unique_count = df[col].nunique()
            sample_values = df[col].unique()[:5]  # Show first 5 unique values
            results.append(f"{col}: {unique_count:,} unique values")
            results.append(f"Sample values: {', '.join(str(x) for x in sample_values)}")
        return "\n".join(results)
        
    else:
        return (
            "I understand you're asking about the data. Here are the types of analysis I can perform:\n"
            "- Statistical: mean, median, standard deviation\n"
            "- Distribution: quantiles, distribution analysis\n"
            "- Temporal: trends over time (for date columns)\n"
            "- Correlation analysis\n"
            "- Category analysis\n"
            "- Missing value analysis\n"
            "- Summary statistics\n"
            "- Outlier detection\n"
            "- Count and unique value analysis\n"
            "\nPlease rephrase your question using these keywords."
        )

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        print(f"Starting {SERVER_NAME}...")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )