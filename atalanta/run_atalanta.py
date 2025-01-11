import numpy as np
import pandas as pd
import os
import dask.dataframe as dd
import csv
from tabulate import tabulate


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


    def get_probability_tables(path):
        csv_paths = []
        pt_dict = dict()
        # Ensure the path exists
        if os.path.exists(path):
            # Get list of all files and directories in the specified path
            files_and_dirs = os.listdir(path)

            # Filter out directories to get only files
            csv_paths = [f for f in files_and_dirs if os.path.isfile(os.path.join(path, f))]

        else:
            print(f"The specified path '{path}' does not exist.")
        for csv_path in csv_paths:
            pt_dict[filename_to_key(csv_path)] = csv_to_dict(os.path.join(path, csv_path))
        return pt_dict

    def run_quantization(input_array):
        # Handle non-finite values
        input_array = np.nan_to_num(input_array, nan=0, posinf=255, neginf=0)

        # Normalize to 0-255 range and convert to uint8
        input_array = ((input_array - input_array.min()) / (input_array.max() - input_array.min()) * 255).astype(np.uint8)

        return input_array

    def add_row_to_csv(row, output_file):
        # Append the row to the file
        with open(output_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Model_Name', 'Layer', 'Type', 'Symbol_Stream', 'Offset_Stream', 'Offset_Length_Stream'])
            writer.writerow(row)

    def run_atalanta(input_stream, prob_table):
        # Initialize the encoder
        encoder = AtalantaEncoder(prob_table)

        # Run the encoder
        encoder.encode(input_stream.tolist())

        # Finalize the encoding process
        symbol_stream, offset_stream, offset_length_stream = encoder.finalize()

        return symbol_stream, offset_stream, offset_length_stream

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


    # Path to your CSV file
    pt_weights_csv_path = '/content/drive/MyDrive/CSCE_614/Project/probability_table_gen_results/weights_probability_tables'
    pt_act_csv_path = '/content/drive/MyDrive/CSCE_614/Project/probability_table_gen_results/activations_probability_tables'

    weights_csv_path = '/content/drive/MyDrive/CSCE_614/Project/weights_all_layers.csv'
    act_csv_path = '/content/drive/MyDrive/CSCE_614/Project/activations_all_layers.csv'

    #results_output_directory = '/content/drive/MyDrive/CSCE_614/Project/atalanta_outputs'
    # Create the output directory if it doesn't exist
    #os.makedirs(os.path.dirname(results_output_directory), exist_ok=True)

    # Output file paths
    weights_encoded_output_file = '/content/drive/MyDrive/CSCE_614/Project/atalanta_outputs/atalanta_encoded_output_weights.csv'
    act_encoded_output_file = '/content/drive/MyDrive/CSCE_614/Project/atalanta_outputs/atalanta_encoded_output_activations.csv'


    # Output CSV file paths
    weights_summary_file = '/content/drive/MyDrive/CSCE_614/Project/atalanta_outputs/atalanta_encoded_summary_weights.csv'
    act_summary_file = '/content/drive/MyDrive/CSCE_614/Project/atalanta_outputs/atalanta_encoded_summary_activations.csv'


    

    file_path_dict = {
    'weights' : {'pt_tables': pt_weights_csv_path , 'input_stream': weights_csv_path, 'encoded_output': weights_encoded_output_file, 'encoded_summary': weights_summary_file},
    'activaitions' : {'pt_tables': pt_act_csv_path , 'input_stream': act_csv_path, 'encoded_output': act_encoded_output_file, 'encoded_summary': act_summary_file},
    }

    for vtype in file_path_dict.keys():
        pt_csv_path = file_path_dict[vtype]['pt_tables']
        values_csv_path = file_path_dict[vtype]['input_stream']
        encoded_output_file = file_path_dict[vtype]['encoded_output']
        csv_summary_file = file_path_dict[vtype]['encoded_summary']

        # Write the header row (only once)
        with open(encoded_output_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Model_Name', 'Layer', 'Type', 'Symbol_Stream', 'Offset_Stream', 'Offset_Length_Stream'])
            writer.writeheader()

        # Get the probability tables
        probability_tables = get_probability_tables(pt_csv_path)
        #print(probability_tables.keys())

        # Read the weights and activaitions CSV file
        #input_df = pd.read_csv(weights_csv_path)

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


                    # Get the probability table
                    pt_file_name = f"{row['Model Name']}_{row['Layer Number']}_{row['Type']}"
                    prob_table = probability_tables[pt_file_name]

                    # encode using Atalanta Encoder
                    symbol_stream, offset_stream, offset_length_stream = run_atalanta(input_array,prob_table)

                    output_row = {
                        'Model_Name': row['Model Name'],
                        'Layer': row['Layer Number'],
                        'Type': row['Type'],
                        'Symbol_Stream': symbol_stream,
                        'Offset_Stream': offset_stream,
                        'Offset_Length_Stream': offset_length_stream
                    }

                    # Append the row to the CSV file
                    add_row_to_csv(output_row, encoded_output_file)

                    input_stream_length = len(input_array)
                    input_stream_length_bits = input_stream_length*8
                    symbol_stream_length = len(symbol_stream)
                    offset_length_stream_length = sum(offset_length_stream)
                    compression_ratio = (input_stream_length_bits)/(symbol_stream_length + offset_length_stream_length)
                    compression_percentage = (1-(1/compression_ratio))*100

                    output_summary = {
                        'Model_Name': row['Model Name'],
                        'Layer_Number': row['Layer Number'],
                        'Type': row['Type'],
                        'Input_Stream_Length (values)': input_stream_length,
                        'Original_Length (bits)': input_stream_length_bits,
                        'Symbol_Stream_Length (bits)': symbol_stream_length,
                        'Offset_Stream_Length (bits)': offset_length_stream_length,
                        'After Compression (bits)': (symbol_stream_length + offset_length_stream_length),
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
                        'After Compression (bits)': (symbol_stream_length + offset_length_stream_length),
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
