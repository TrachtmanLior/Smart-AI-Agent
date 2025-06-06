from elasticsearch import Elasticsearch
from config import dataset_type


# Elasticsearch connection (for logging alerts)
if dataset_type == 'elastic_flows':
    es = Elasticsearch("http://localhost:9200")
    alerts_index = "anomaly_alerts"  # Define your index for anomaly alerts

# Function to check anomalies and send alerts
def check_anomalies(graph):
    for node in graph.nodes:
        # Check anomaly conditions
        graph.nodes[node]["pred"] = graph.nodes[node]["cluster_pred"] or (graph.nodes[node]["ann_pred"] and graph.nodes[node]["cluster"] == -1)
        
        if graph.nodes[node]["pred"] and not graph.nodes[node]["printed"]:  # If the entity is detected as an anomaly
            ip = graph.nodes[node]["ip"].split("_")[0]
            port = graph.nodes[node]["port"]
            anomaly_message = f'The entity {ip}:{port} is anomaly'
            
            if dataset_type == 'elastic_flows':
                # Create the alert document
                alert_doc = {
                    "ip": ip,
                    "port": port,
                    "anomaly_message": anomaly_message,
                    "timestamp": graph.nodes[node]["packet_index"],
                    "alert_type": "Anomaly",
                    "status": "Triggered",
                    "node_info": graph.nodes[node]  # Additional node info can be included
                }
                
                # Insert anomaly alert into Elasticsearch
                es.index(index=alerts_index, body=alert_doc)
                print(f"Alert logged in Elasticsearch for {ip}:{port}")
            
            else:
                print(anomaly_message)
            
            graph.nodes[node]["printed"] = True

