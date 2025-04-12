import lasio
import pandas as pd

def read_las_file(file_path):
    """
    Reads a LAS file and returns a DataFrame with the data.

    Args:
        file_path (str): Path to the LAS file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the LAS file.
    """
    # Read the LAS file
    las = lasio.read(file_path)

    # Convert to DataFrame
    df = las.df()

    # Return the DataFrame
    return df

# Testing the function
read_las_file("../las-files/T10746d80.las")
print(read_las_file("las-files/T10746d80.las"))
