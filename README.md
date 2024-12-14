# Data Parser MCP Server

A Model Context Protocol (MCP) server for loading, analyzing, and visualizing CSV data. This server enables loading CSV files into memory, performing various types of data analysis through natural language queries, and creating data visualizations.

## Features

### Data Loading
- Load CSV files with configurable options
- Automatic detection of column types
- Support for date column parsing
- Metadata storage for loaded datasets

### Analysis Capabilities

#### Statistical Analysis
- Mean/average calculations
- Median calculations
- Standard deviation
- Quantile/percentile analysis
- Distribution analysis

#### Temporal Analysis
- Time series analysis
- Trend detection
- Monthly averages for date columns

#### Correlation Analysis
- Correlation matrices for numerical columns
- Relationship detection between variables

#### Categorical Analysis
- Value counts and distributions
- Category frequency analysis

#### Data Quality Analysis
- Missing value detection
- Outlier detection using IQR method
- Data completeness reporting

### Visualization Capabilities

#### Chart Types
- Bar charts
- Line charts
- Scatter plots
- Pie charts
- Heatmaps
- Box plots
- Histograms

#### Visualization Features
- Group-by functionality for aggregated visualizations
- Multiple column support
- Customizable options (bins, titles, etc.)
- JSON-based chart data format for easy integration

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd data-parser
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Configuration

Add the server to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "mcp-server-data-parser": {
      "command": "/Path/to/.local/bin/uv",
      "args": [
        "--directory",
        "path/to/directory",
        "run",
        "mcp-server-data-parser"
      ]
    }
  }
}
```

### Available Tools

#### 1. load-csv
Loads a CSV file into memory for analysis.

Parameters:
- `file_path`: Path to the CSV file
- `dataset_name`: Name to reference this dataset
- `options`: (Optional)
  - `separator`: CSV separator (default: ",")
  - `encoding`: File encoding (default: "utf-8")
  - `skiprows`: Number of rows to skip (default: 0)
  - `date_columns`: List of column names to parse as dates

Example:
```python
await call_tool("load-csv", {
    "file_path": "data.csv",
    "dataset_name": "my_data",
    "options": {
        "date_columns": ["transaction_date"]
    }
})
```

#### 2. analyze-data
Analyzes loaded dataset based on natural language questions.

Parameters:
- `dataset_name`: Name of the dataset to analyze
- `question`: Question about the data to analyze
- `columns`: (Optional) Specific columns to analyze
- `group_by`: (Optional) Column to group by for analysis

Example:
```python
await call_tool("analyze-data", {
    "dataset_name": "my_data",
    "question": "What is the correlation between numeric columns?",
    "columns": ["price", "quantity", "total"],
    "group_by": "category"
})
```

#### 3. visualize-data
Creates various types of data visualizations.

Parameters:
- `dataset_name`: Name of the dataset to visualize
- `visualization_type`: Type of visualization to create
  - Options: "bar", "line", "scatter", "pie", "heatmap", "boxplot", "histogram"
- `columns`: List of columns to include in visualization
- `group_by`: (Optional) Column to group by
- `options`: (Optional)
  - `bins`: Number of bins for histogram
  - `title`: Chart title

Examples:
```python
# Bar chart
await call_tool("visualize-data", {
    "dataset_name": "sales_data",
    "visualization_type": "bar",
    "columns": ["revenue"],
    "group_by": "region"
})

# Scatter plot
await call_tool("visualize-data", {
    "dataset_name": "sales_data",
    "visualization_type": "scatter",
    "columns": ["price", "quantity"]
})

# Histogram
await call_tool("visualize-data", {
    "dataset_name": "sales_data",
    "visualization_type": "histogram",
    "columns": ["revenue"],
    "options": {
        "bins": 30,
        "title": "Revenue Distribution"
    }
})
```

### Visualization Types and Use Cases

1. **Bar Charts**
   - Compare values across categories
   - Show distributions of categorical data
   - Display aggregated values by group

2. **Line Charts**
   - Show trends over time
   - Track changes in metrics
   - Compare multiple series

3. **Scatter Plots**
   - Identify correlations between variables
   - Spot patterns and clusters
   - Detect outliers

4. **Pie Charts**
   - Show composition of a whole
   - Display percentage distributions
   - Compare parts of a total

5. **Heatmaps**
   - Visualize correlation matrices
   - Show patterns in dense data
   - Display cross-tabulations

6. **Box Plots**
   - Show distribution characteristics
   - Identify outliers
   - Compare distributions across groups

7. **Histograms**
   - View data distributions
   - Identify patterns and skewness
   - Check for normality

### Supported Analysis Questions

The server understands various types of analysis questions including:

1. Statistical Analysis
   - "What is the mean of the numeric columns?"
   - "Show me the median values"
   - "Calculate the standard deviation"

2. Distribution Analysis
   - "Show me the distribution of values"
   - "What are the quantiles?"
   - "Show me the percentiles"

3. Temporal Analysis
   - "Show me the trend over time"
   - "What is the time series pattern?"
   - "How does it change over time?"

4. Correlation Analysis
   - "What is the correlation between columns?"
   - "Show me how variables correlate"

5. Category Analysis
   - "Show me category distributions"
   - "What are the categorical counts?"

6. Data Quality
   - "Show me missing values"
   - "Find outliers in the data"
   - "How many null values are there?"

## Dependencies

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- MCP SDK

## Development

To contribute or modify the server:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

[Your chosen license]

## Contact

[Your contact information]