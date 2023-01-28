from pandas import DataFrame
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """

    df['text'] = df['title'] + df['description']
    df = df[['text','tags','like_count']]
    df = df.dropna(axis= 0, how= 'any')

    df["tags_1"]= df["tags"].apply(lambda x: x[1:-1])

    df['text']= df['text'] + df["tags_1"]

    df["like_count_1"] = pd.qcut(df["like_count"], 2, labels=[0,1]).astype("int64")

    df = df.drop(['tags','like_count'], axis=1)

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
