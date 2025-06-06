from elasticsearch import Elasticsearch
import time
from tri_graph import TriGraph
from network import ANN
from results import measure_results
from execute_pipeline import execute_pipeline

def process_real_time_flows(dic_feature_to_name, index_name, num_of_flows=1000, poll_interval=5, algo='clustering', plot=False):
    if algo not in ['ann', 'clustering', 'combined']:
        print("No valid algorithm specified.")
        return

    es = Elasticsearch("http://localhost:9200")  
    tri_graph = TriGraph()
    pred = []
    label = []
    node_to_index = {}

    last_timestamp = 0  
    eof_detected = False  # Flag to track termination signal

    print("Listening for new flows in Elasticsearch...")
    
    while True:
        # Check if termination signal exists
        if es.exists(index=index_name, id="EOF"):
            print("Termination signal detected. Processing remaining flows...")
            eof_detected = True  

        # Query new flows since the last timestamp
        query = {
            "query": {
                "range": {
                    "Timestamp": {"gt": last_timestamp}  
                }
            },
            "sort": [{"Timestamp": "asc"}],  
            "size": num_of_flows  
        }

        response = es.search(index=index_name, body=query)
        hits = response["hits"]["hits"]

        if not hits:
            if eof_detected:
                print("No more flows left to process. Exiting...")
                break  
            else:
                print("No new flows detected. Waiting...")
                time.sleep(poll_interval)
                continue  

        print(f"Processing {len(hits)} new flows...")

        for hit in hits:
            row = hit["_source"]

            if row[dic_feature_to_name['Protocol']] != dic_feature_to_name['TCP']:
                continue  

            tri_graph.add_flow_to_graph(row, pred, label, node_to_index, dic_feature_to_name)

        last_timestamp = hits[-1]["_source"]["Timestamp"]

        execute_pipeline(tri_graph, algo, plot, pred, node_to_index)

        print("Waiting for new flows...")

        time.sleep(poll_interval)

    measure_results(tri_graph.graph)