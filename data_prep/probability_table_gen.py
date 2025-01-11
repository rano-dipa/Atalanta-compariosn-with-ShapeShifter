import numpy as np
import pandas as pd
import time
import subprocess
import re
import os
import csv

results_path = '/content/drive/MyDrive/CSCE_614/Project/probability_table_gen_results'
values_path = '/content/drive/MyDrive/CSCE_614/Project/'

def run_atalanta(input_array):
    # Handle non-finite values and ensure uint8 conversion
    input_array = np.nan_to_num(input_array, nan=0, posinf=255, neginf=0).astype(np.uint8)

    # Save input array as temporary .npy file
    np.save('temp_input.npy', input_array)

    # Run Atalanta algorithm
    result = subprocess.run(['python', 'atalanta_numpy.py', 'temp_input.npy', '8'], capture_output=True, text=True)
    output = result.stdout

    # Parse output
    data = []
    for line in output.split('\n'):
        match = re.match(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)', line)
        if match:
            data.append([int(match.group(1)), int(match.group(2)), int(match.group(3)),
                         int(match.group(4)), int(match.group(5)), float(match.group(6))])

    # If no data is parsed, return an empty DataFrame
    if len(data) == 0:
        print("No data was parsed from Atalanta output.")
        return pd.DataFrame(columns=['v_min', 'v_max', 'OL', 't_low', 't_high', 'p'])

    # Create and process DataFrame
    columns = ['off', 'v_min', 'abits', 'obits', 'vcnt', 'vcnt/value_cnt']
    df = pd.DataFrame(data, columns=columns)

    df['v_max'] = df['v_min'] + (2 ** df['off'] - 1)
    df['OL'] = np.log2(df['v_max'] - df['v_min'] + 1).round().astype(int)
    value_cnt = df['vcnt'].sum()
    df['p'] = df['vcnt'] / value_cnt

    # Calculate t_low and t_high
    tlow = [0] + list((df['p'].cumsum().iloc[:-1] * 0x3ff + 1).astype(int))
    thigh = (df['p'].cumsum() * 0x3ff).astype(int)
    thigh.iloc[-1] = 0x3ff

    df['t_low'] = tlow
    df['t_high'] = thigh

    # Select final columns
    final_columns = ['v_min', 'v_max', 'OL', 't_low', 't_high', 'p']
    final_df = df[final_columns]

    return final_df

values_dict = {'activations': os.path.join(values_path, 'activations_all_layers.csv'), 'weights': os.path.join(values_path, 'weights_all_layers.csv')}

for type, path in values_dict.items():

    # Create output directories and CSV for timing
    csv_dir = type+'_probability_tables'
    csv_dir = os.path.join(results_path, csv_dir)
    os.makedirs(csv_dir, exist_ok=True)
    timing_results = []

    # Process CSV line by line
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Read header row

        for row in csvreader:
            try:
                # Extract metadata
                model_name = row[0]
                layer_number = row[1]
                row_type = row[2]

                # Extract numeric values after the first three columns
                numeric_values = np.array(row[3:], dtype=np.float32)

                # Track time to generate the table
                start_time = time.time()
                final_df = run_atalanta(numeric_values)
                end_time = time.time()
                time_taken = end_time - start_time

                # If the output DataFrame is empty, skip this row
                if final_df.empty:
                    print(f"Skipping {model_name}, Layer {layer_number}, Type {row_type} due to empty output.")
                    continue

                # Save the generated table to a CSV file
                row_name = f"{model_name}_{layer_number}_{row_type}"
                csv_path = os.path.join(results_path, csv_dir, f'pt_{row_name}.csv')
                final_df.to_csv(csv_path, index=False)

                # Append timing information to the results
                timing_results.append({'Model': model_name, 'Layer': layer_number, 'Type': row_type, 'Time Taken (s)': time_taken})

            except Exception as e:
                print(f"Error processing row {row}: {e}")
                continue

    # Save timing results to a CSV file
    timing_df = pd.DataFrame(timing_results)
    timing_path = os.path.join(results_path, type+'_pt_gen_timing_results.csv')
    timing_df.to_csv(timing_path, index=False)

    # Clean up temporary file
    if os.path.exists('temp_input.npy'):
        os.remove('temp_input.npy')

    print(f"Processing complete. {type} Probability Tables saved in {csv_dir} directory.")
    print(f"Timing results saved to {timing_path}.")

    # Zip the atalanta_tables directory
    subprocess.run(['zip', '-r', os.path.join(results_path, csv_dir+'.zip'), os.path.join(results_path, csv_dir)])