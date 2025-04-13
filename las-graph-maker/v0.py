import lasio
import matplotlib.pyplot as plt
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
    las = lasio.read(file_path, ignore_header_errors=True)

    # Convert to DataFrame
    df = las.df()

    # Return the DataFrame
    return df


def plot_las_data(df):
    """
    Plots the LAS data from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the LAS data.

    Returns:
        None
    """
    # Plot the data
    depth = df.index
    for column in df.columns:
        plt.plot(df[column], depth, label=column)
    # TODO: If the average range of values in a column is larger than around 20% of the other columns,
    # plot in a separate figure. Go through each column and check the range of values.
    # So maybe have a list containing these columns that are too large, then plotting the columns in that list after the
    # main plot has been created.
    plt.gca().invert_yaxis()
    #plt.xlim(-600, 400)
    plt.legend()
    plt.show()

# Testing the function
readf = read_las_file("/home/tikki/Documents/School/BachelorProject/Digitizing-overlapping-curves/las-files/T10746d80.las")
# Dept is y-axis, the other ones are x vals (non rotated so we should prob rotate)
plot_las_data(readf)
