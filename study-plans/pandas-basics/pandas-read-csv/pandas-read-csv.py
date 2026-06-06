import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    df =  pd.DataFrame(data)
    rows, cols = df.shape
    columns = df.columns.to_list()

    return {'data': df.to_dict('list'),
           'shape': [rows, cols],
           'columns': columns
    }