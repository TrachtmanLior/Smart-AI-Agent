import sys

from flow_separation import separate_packets_pcap
from flow_separation_csv import separate_packets_csv
from csv_reading import process_flows
from process_flows_elastic import process_real_time_flows
from config import dataset_type
from config import feature_to_name
# from config import index_name
import time
import psutil
import os

if __name__ == '__main__':

    # Check for command-line argument input_file_path
    if len(sys.argv) < 2:
        print('usage: main.py input_file_path num_of_flows')
        exit(1)
    input_file_path = sys.argv[1]

    # Check for command-line argument num_of_flows
    if len(sys.argv) < 3 or not sys.argv[2].isdecimal():
        num_of_flows = 2000
    else:
        num_of_flows = int(sys.argv[2])
    
    # Get current process
    process = psutil.Process(os.getpid())

    # Start time
    start_time = time.time()

    if dataset_type == 'packets_csv':
        separate_packets_csv(input_file_path, num_of_rows=-1, algo='clustering', plot=False, num_of_flows=num_of_flows)
    elif dataset_type == 'packets_pcap':
        separate_packets_pcap(input_file_path, num_of_rows=-1, algo='clustering', plot=False, num_of_flows=num_of_flows)
    elif dataset_type == 'elastic_flows':
        process_real_time_flows(feature_to_name, input_file_path, num_of_flows, poll_interval=3, algo='combined', plot=False)
    else: #labeled_data
        process_flows(feature_to_name, input_file_path, num_of_flows, num_of_rows=-1, algo='combined', plot=False)
        
    # End time
    end_time = time.time()

    # Processing time
    processing_time = end_time - start_time

    # CPU and memory usage
    cpu_usage = process.cpu_percent(interval=1)  # % over 1 sec
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB

    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {memory_usage:.2f} MB")
