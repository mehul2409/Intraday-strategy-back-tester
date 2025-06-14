# convert_csv_to_parquet.py
import polars as pl
from pathlib import Path
import time

def process_csvs_to_individual_parquets():
    """
    Reads each CSV file from the 'csv_data' directory and saves it
    as its own individual, compressed Parquet file.
    """
    start_time = time.time()

    # Define the input and output paths
    csv_folder = Path("old_data")
    output_folder = Path("old_parquet_data")
    output_folder.mkdir(exist_ok=True) # Create the output folder if it doesn't exist

    # 1. Find all CSV files in the input folder
    csv_files = list(csv_folder.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the 'csv_data' directory. Please check the path.")
        return

    print(f"Found {len(csv_files)} CSV files to convert...")
    
    files_converted = 0
    # 2. Loop through each CSV file and convert it individually
    for file_path in csv_files:
        try:
            # Determine the output filename
            symbol = file_path.stem.upper()
            output_file_path = output_folder / f"{symbol}.parquet"
            
            # Read the CSV file
            # Assuming columns are like: date,open,high,low,close,volume
            df = pl.read_csv(
                file_path,
                try_parse_dates=True
            )
            
            # Write the DataFrame directly to its own Parquet file
            df.write_parquet(
                output_file_path,
                compression='snappy'  # Use Snappy compression for Parquet files - efficient and fastest
            )
            
            print(f"Successfully converted: {file_path.name} -> {output_file_path.name}")
            files_converted += 1

        except Exception as e:
            print(f"Could not process file {file_path}. Error: {e}")

    end_time = time.time()
    print("\n--- Process Complete! ---")
    print(f"Total files converted: {files_converted}/{len(csv_files)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Parquet files are saved in the '{output_folder}' directory.")


if __name__ == "__main__":
    process_csvs_to_individual_parquets()