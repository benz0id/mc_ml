import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Iterable
import pandas as pd
import torch

from data_prep.world_parsing.block_id_mapping import BlockIDMapper


def create_pie_chart(data: pd.DataFrame,
                     threshold: Optional[float] = 1.0) -> None:
    """
    Creates a pie chart from a DataFrame with two columns: labels and values.
    If any category's percentage is smaller than the threshold, it is aggregated
    into an "Other" category.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing two columns:
        the first column should be the category names (str),
        the second column should be the corresponding numerical values (int).
    threshold : float, optional
        The percentage threshold below which categories are grouped into "Other".
        Defaults to 1.0 (1%).

    Raises
    ------
    ValueError
        If the DataFrame does not have exactly two columns, or if the data
        types are not as expected.

    ## GPT4
    """
    # Defensive programming: Check that the DataFrame has exactly two columns
    if data.shape[1] != 2:
        raise ValueError("DataFrame must have exactly two columns.")

    # Check that the first column is of type 'str' and the second is 'int' or 'float'
    if not pd.api.types.is_string_dtype(data.iloc[:, 0]):
        raise ValueError("First column must contain string labels.")
    if not pd.api.types.is_numeric_dtype(data.iloc[:, 1]):
        raise ValueError("Second column must contain numeric values.")

    # Extract the labels and values
    labels = data.iloc[:, 0]
    values = data.iloc[:, 1]

    # Calculate total sum of values
    total_sum = values.sum()

    # Calculate the percentage of each category
    percentages = (values / total_sum) * 100

    # Aggregate categories below the threshold into "Other"
    filtered_data = data.copy()
    other_value = filtered_data[percentages < threshold].iloc[:, 1].sum()
    filtered_data = filtered_data[percentages >= threshold]

    if other_value > 0:
        # Append the "Other" category
        filtered_data = pd.concat([
            filtered_data,
            pd.DataFrame([['Other', other_value]], columns=data.columns)
        ], ignore_index=True)

    # Recalculate labels and values after filtering
    labels = filtered_data.iloc[:, 0]
    values = filtered_data.iloc[:, 1]

    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
           counterclock=False,
           wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

    # Ensure the pie is a circle
    ax.axis('equal')

    # Set a title for clarity
    plt.title("Distribution of Categories", fontsize=14)

    # Display the chart
    plt.show()



def get_block_counts(chunks: Iterable[torch.tensor], mapper: BlockIDMapper) -> pd.DataFrame:
    """
    Provides the number of blocks in the given chunks.

    chunks
        An iterable list of chunk tensors.
    mapper
        The mapper used to encode the blocks.

    :return: block - count dataframe.
    """
    max_val = mapper.get_max_id() + 1
    counts = torch.zeros(max_val, dtype=torch.int32)
    for chunk in chunks:
        counts += chunk.reshape(-1).bincount(minlength=max_val)

    block_counts = {}
    for i in range(len(counts)):
        item = mapper.get_block(i)
        if item.id in block_counts:
            block_counts[item.id] += counts[i].item()
        else:
            block_counts[item.id] = counts[i].item()

    df = (pd.DataFrame(block_counts.items())
          .sort_values(by=1, ascending=False)
          .reset_index(drop=True))
    df.columns = ['block', 'count']

    return df
