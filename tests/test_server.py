import pytest
from mcp_server_data_parser.server import create_visualization, datasets
import pandas as pd
import numpy as np

@pytest.fixture
def sample_dataset():
    df = pd.DataFrame({
        'x': range(10),
        'y': np.random.rand(10),
        'category': ['A', 'B'] * 5
    })
    datasets['test_viz'] = df
    return 'test_viz'

@pytest.mark.asyncio
async def test_create_visualization(sample_dataset):
    # Test bar chart
    result = await create_visualization({
        "dataset_name": sample_dataset,
        "visualization_type": "bar",
        "columns": ["x", "y"],
        "group_by": "x"
    })
    
    assert len(result) == 1
    assert result[0].type == "text"
    assert "Visualization type: bar" in result[0].text
    
    # Test with invalid dataset
    with pytest.raises(ValueError, match="Dataset 'invalid' not found"):
        await create_visualization({
            "dataset_name": "invalid",
            "visualization_type": "bar",
            "columns": ["x", "y"],
            "group_by": "x"
        })
    
    # Test with missing required parameters
    with pytest.raises(ValueError, match="Both dataset_name and visualization_type are required"):
        await create_visualization({
            "dataset_name": sample_dataset
        })