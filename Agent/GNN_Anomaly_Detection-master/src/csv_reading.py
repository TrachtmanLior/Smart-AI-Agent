import csv

from tri_graph import TriGraph
from network import ANN
from results import measure_results
from execute_pipeline import execute_pipeline

def process_flows(dic_feature_to_name, input_file_path=None, num_of_flows=None, num_of_rows=-1, algo='clustering', plot=False):
                
    if algo in ['ann', 'clustering', 'combined']:
        tri_graph = TriGraph()
    
    else:
        print("No valid algorithm specified.")
        return
        
    with open(input_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        pred = []
        label = []
        node_to_index = {}
        
        # Iterate through each line in the CSV
        for i, row in enumerate(csv_reader):
            if i % 10000 == 0:
                print(f'processed {i} flows')
            
            if i == num_of_rows:
                break
            
            if row[dic_feature_to_name['Protocol']] != dic_feature_to_name['TCP']:
                continue
            
            tri_graph.add_flow_to_graph(row, pred, label, node_to_index, dic_feature_to_name)

            if i and i % num_of_flows == 0:
                execute_pipeline(tri_graph, algo, plot, pred, node_to_index)
                
        execute_pipeline(tri_graph, algo, plot, pred, node_to_index)
                
        measure_results(tri_graph.graph)