import os
from os import walk

import lasio
import matplotlib.pyplot as plt
import numpy as np

odd_column_keeper = []
file_list = []
file_watch = []
path =  "/home/tikki/Documents/School/BachelorProject/Digitizing-overlapping-curves/las-files"
dest =  "/home/tikki/Documents/School/BachelorProject/Digitizing-overlapping-curves/las-graph-maker/assets"
if not os.path.exists(dest):
    os.makedirs(dest)

def file_finder(path):
    for root, dirc, files in walk(path):
        for fileName in files:
            file_list.append(fileName)
#file_finder(path)
#print(file_list)

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


def plot_las_data(df, file):
    """
    Plots the LAS data from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the LAS data.

    Returns:
        None
    """
    file_path = os.path.join(dest, file)
    # Plot the data
    depth = df.index
    min_depth = df.index.min()
    max_depth = df.index.max()

    # Create ticks at 100 intervals
    tick_start = int(min_depth)
    tick_end = int(max_depth)
    depth_ticks = np.arange(tick_start, tick_end + 50, 50)

    # Create a tall figure
    plt.figure().set_figheight(100)

    for column in df.columns:
        #odd_column_keeper.append(df[column].max() < 1000)
        if df[column].max() < 1000:
            plt.plot(df[column], depth, label=column, color='black')
        elif df[column].max() > 1000:
            odd_column_keeper.append(column)
    plt.gca().invert_yaxis()
    #plt.xlim(-400, 600)
    plt.yticks(depth_ticks)
    # plt.legend()
    plt.savefig(f"{file_path}-1.tif", bbox_inches='tight', dpi=200)
    plt.close()
    #plt.show()

def plot_odd_curves(lst, df, file):
    file_path = os.path.join(dest, file)
    depth = df.index
    plt.figure().set_figheight(100)
    for column in df.columns:
        if column in odd_column_keeper:
            plt.plot(df[column], depth, label=column, color='black')
    plt.gca().invert_yaxis()
    # plt.legend()
    plt.savefig(f"{file_path}-2.tif", bbox_inches='tight', dpi=200)
    plt.close()
    #plt.show()

def process_files(path, files):
    file_finder(path)
    for fileName in file_list:
        try:
            df = read_las_file(f"{path}/{fileName}")

            plot_las_data(df, fileName)

            plot_odd_curves(odd_column_keeper, df, fileName)

            odd_column_keeper.clear()
        except KeyError as e:
            if "No ~ sections found. Is this a LAS file?" in str(e):
                file_watch.append(fileName)
                print(f"{fileName} added to watch list")
                continue
        except Exception as e:
            print(f"Exception at {fileName}: {str(e)}")
            continue
        if file_watch:
            print("\nFiles that couldn't be processed:")
            for file in file_watch:
                print(file)


# Testing the function
#readf = read_las_file("/home/tikki/Documents/School/BachelorProject/Digitizing-overlapping-curves/las-files/T10746d80.las")
# Dept is y-axis, the other ones are x vals (non rotated so we should prob rotate)
#plot_las_data(readf)
#plot_odd_curves(odd_column_keeper, readf)
#print(odd_column_keeper)
process_files(path, file_list)
