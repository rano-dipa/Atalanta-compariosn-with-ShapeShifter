import numpy as np
import pandas as pd
import os
import dask.dataframe as dd
import csv
from tabulate import tabulate
import numpy as np
import random
import time


def main():

    def filename_to_key(filename):
        # Remove the prefix and suffix
        base_name = filename.removeprefix("pt_").removesuffix(".csv")
        # Split the remaining part into components
        #parts = base_name.split("_")
        return base_name

    def csv_to_dict(csv_path):
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_path)
        # Convert the DataFrame to a list of dictionaries
        data_dict = df.to_dict(orient='records')
        return data_dict

    def add_row_to_csv(row, output_file):
        # Append the row to the file
        with open(output_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Model_Name', 'Layer', 'Type', 'Encoded_Stream'])
            writer.writerow(row)

    def print_encoded_summary_table(summary_table):

        # Convert rows to tabulate format
        table = [list(row.values()) for row in summary_table]
        headers = summary_table[0].keys()  # Use keys of the first row as headers

        # Pretty-print the table
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def output_summary_to_csv(csv_table, output_file):

        # Write to the CSV file
        with open(output_file, mode='w', newline='') as file:
            # Create a CSV DictWriter object
            writer = csv.DictWriter(file, fieldnames=csv_table[0].keys())

            # Write the header row
            writer.writeheader()

            # Write each row
            for row in csv_table:
                writer.writerow(row)

        print(f"Data has been written to {output_file}")

    # ShapeShifter encoding function
    def shapeshifter_encode(data, group_size=16):
        encoded_data = []
        encoded_size = 0 
        for i in range(0, len(data), group_size):
            group = data[i:i+group_size]
            max_val = np.max(np.abs(group))
            bits_needed = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
            encoded_data.append((bits_needed, group.tolist()))
            encoded_size += bits_needed * len(group)
        return encoded_data, encoded_size



    weights_csv_path = '/content/drive/MyDrive/CSCE_614/Project/weights_all_layers.csv'
    act_csv_path = '/content/drive/MyDrive/CSCE_614/Project/activations_all_layers.csv'

    # Output file paths
    weights_encoded_output_file = '/content/drive/MyDrive/CSCE_614/Project/shapeshifter_outputs/shapeshifter_encoded_output_weights.csv'
    act_encoded_output_file = '/content/drive/MyDrive/CSCE_614/Project/shapeshifter_outputs/shapeshifter_encoded_output_activations.csv'

    # Output CSV file paths
    weights_summary_file = '/content/drive/MyDrive/CSCE_614/Project/shapeshifter_outputs/shapeshifter_encoded_summary_weights.csv'
    act_summary_file = '/content/drive/MyDrive/CSCE_614/Project/shapeshifter_outputs/shapeshifter_encoded_summary_activations.csv'

    file_path_dict = {
    'weights' : {'input_stream': weights_csv_path, 'encoded_output': weights_encoded_output_file, 'encoded_summary': weights_summary_file},
    'activaitions' : {'input_stream': act_csv_path, 'encoded_output': act_encoded_output_file, 'encoded_summary': act_summary_file},
    }

    for vtype in file_path_dict.keys():
        values_csv_path = file_path_dict[vtype]['input_stream']
        encoded_output_file = file_path_dict[vtype]['encoded_output']
        csv_summary_file = file_path_dict[vtype]['encoded_summary']

        # Write the header row (only once)
        with open(encoded_output_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Model_Name', 'Layer', 'Type', 'Encoded_Stream'])
            writer.writeheader()
        
        summary_table = []
        csv_file_out = []

        # Process CSV line by line
        with open(values_csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)  # Read header row

            for row_data in csvreader:
                try:
                    row = {'Model Name':row_data[0] , 'Layer Number':row_data[1], 'Type':row_data[2]}

                    # Extract numeric values after the first three columns
                    input_array = np.array(row_data[3:], dtype=np.uint8)


                    # encode using Atalanta Encoder
                    encoded_stream, encoded_size = shapeshifter_encode(input_array)

                    output_row = {
                        'Model_Name': row['Model Name'],
                        'Layer': row['Layer Number'],
                        'Type': row['Type'],
                        'Encoded_Stream': encoded_stream,
                    }

                    # Append the row to the CSV file
                    add_row_to_csv(output_row, encoded_output_file)

                    input_stream_length = len(input_array)
                    input_stream_length_bits = input_stream_length*8
                    encoded_stream_length = encoded_size
                    compression_ratio = (input_stream_length_bits)/encoded_stream_length
                    compression_percentage = (1-(1/compression_ratio))*100



                    output_summary = {
                        'Model_Name': row['Model Name'],
                        'Layer_Number': row['Layer Number'],
                        'Type': row['Type'],
                        'Input_Stream_Length (values)': input_stream_length,
                        'Original_Length (bits)': input_stream_length_bits,
                        'After Compression (bits)': encoded_stream_length,
                        'Compression_Ratio': compression_ratio,
                        'Compression_Percentage': compression_percentage
                        }

                    summary_table.append(output_summary)

                    csv_summary = {
                        'Model_Name': row['Model Name'],
                        'Layer_Number': row['Layer Number'],
                        'Type': row['Type'],
                        'Input_Stream_Length (values)': input_stream_length,
                        'Original (bits)': input_stream_length_bits,
                        'After Compression (bits)': encoded_stream_length,
                        'Compression_Ratio': compression_ratio,
                        'Compression_Percentage': compression_percentage
                        }

                    csv_file_out.append(csv_summary)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue


        # Print the summary table
        print_encoded_summary_table(summary_table)
        output_summary_to_csv(csv_file_out, csv_summary_file)

if __name__ == "__main__":
    main()
