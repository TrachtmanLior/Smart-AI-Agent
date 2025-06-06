import networkx as nx

from vector import Vector
from config import attacker_ip, victom_ip, dataset_type

colors = ["lightskyblue"]

# Define a class to represent a tri-graph structure for network traffic analysis
class TriGraph():
    def __init__(self) -> None:
        self.ip_to_id = {}
        self.graph = nx.Graph()
        self.count_flows = 1
        # self.ip_to_color = {}
        self.gcn_model = None
    
    from graph_embedding import create_embeddings
    from visualization import visualize_directed_graph
    
    # Generate id number for ip:port
    def get_id(self, key):
        if key not in self.ip_to_id:
            self.ip_to_id[key] = f'{len(self.ip_to_id)}ip'
        return self.ip_to_id[key]

    # Generate color number for ip
    def get_color(self, ip):
        if ip not in self.ip_to_color:
            self.ip_to_color[ip] = colors[len(self.ip_to_color)%len(colors)]
        return self.ip_to_color[ip]

    def add_separated_flow_to_graph(self, vector: Vector):
        # update count_flows
        self.count_flows += 1
        
        # define colors
        src_ip, dst_ip = vector.src.split(":")[0], vector.dst.split(":")[0]
        src_color, dst_color = self.get_color(src_ip), self.get_color(dst_ip)
        
        # check label
        src_label = src_ip in [attacker_ip, victom_ip]
        dst_label = dst_ip in [attacker_ip, victom_ip]
        label = src_label and dst_label
        if dataset_type == 'cic2018':
            src_label, dst_label = label, label
        src_color = "lightcoral" if src_label else src_color
        dst_color = "lightcoral" if dst_label else dst_color
        
        #  add nodes
        src_id, dst_id = self.get_id(vector.src), self.get_id(vector.dst)
        
        if not self.graph.has_node(src_ip):
            self.graph.add_node(src_ip, side = 'Client-IP', amount = 0, length = 0, time_delta = 0.0, 
                                min_packet_length = 0, max_packet_length = 0, mean_packet_length = 0,
                                FIN_count = 0,  SYN_count = 0,  RST_count = 0,  PSH_count = 0,  ACK_count = 0,  
                                URG_count = 0, count_opened_sockets = 0, 
                                anomaly_score_history =  [], cluster = -1,
                                pred = False, label = src_label, cluster_pred = False, ann_pred = False,
                                ip = vector.src, flows = 1, color = src_color, timestamp=vector.timestamp)
                
        if not self.graph.has_node(src_id):
            self.graph.add_node(src_id, side = 'Client', amount = 0, length = 0, time_delta = 0.0, 
                                min_packet_length = 0, max_packet_length = 0, mean_packet_length = 0,
                                FIN_count = 0,  SYN_count = 0,  RST_count = 0,  PSH_count = 0,  ACK_count = 0,  
                                URG_count = 0, count_opened_sockets = 0, 
                                anomaly_score_history =  [], cluster = -1,
                                pred = False, label = src_label, cluster_pred = False, ann_pred = False,
                                ip = vector.src, flows = 0, color = src_color, timestamp=vector.timestamp)

        if not self.graph.has_node(dst_id):
            self.graph.add_node(dst_id, side = 'Server', amount = 0, length = 0, time_delta = 0.0, 
                                min_packet_length = 0, max_packet_length = 0, mean_packet_length = 0,
                                FIN_count = 0,  SYN_count = 0,  RST_count = 0,  PSH_count = 0,  ACK_count = 0,  
                                URG_count = 0, count_opened_sockets = 0,
                                anomaly_score_history =  [], cluster = -1,
                                pred = False, label = dst_label, cluster_pred = False, ann_pred = False,
                                ip = vector.dst, sip = vector.src, flows = 0, color = dst_color, timestamp=vector.timestamp)
        
        # add edges
        if not self.graph.has_edge(src_ip, src_id):
            self.graph.add_edge(src_ip, src_id)
        if not self.graph.has_edge(src_id, dst_id):
            self.graph.add_edge(src_id, dst_id)
        
        # update nodes features
        self.update_features_of_separated_flow(src_id, vector, 'fwd')
        self.update_features_of_separated_flow(src_ip, vector, 'fwd')
        self.update_features_of_separated_flow(dst_id, vector, 'bwd')
        
    def update_features_of_separated_flow(self, id, vector, direction):
        # Update features of a node in the graph
        node = self.graph.nodes[id]
        
        node["flows"] += 1
        node['amount'] += getattr(vector, f'{direction}_packets_amount')
        node['length'] += getattr(vector, f'{direction}_packets_length')
        node['min_packet_length'] += getattr(vector, f'min_{direction}_packet')
        node['max_packet_length'] += getattr(vector, f'max_{direction}_packet')
        
        if node['side'] == 'Client-IP':
            node['count_opened_sockets'] = 0
            node["flows"] = self.graph.degree(id)
            for neighbor in self.graph.neighbors(id):
                node['count_opened_sockets'] += self.graph.nodes[neighbor]["count_opened_sockets"]
        elif vector.state == 'CLOSED':
            node['count_opened_sockets'] = 0
        else:
            node['count_opened_sockets'] = 1
            
        if  node['amount'] != 0:     
            node['mean_packet_length'] = node['length']/node['amount']
        node["time_delta"] +=  vector.time_delta
        node["packet_index"] = vector.packet_index
        
        for flag in vector.flags:
            node[f'{flag}_count'] += vector.flags[flag]
        
        
    def add_flow_to_graph(self, row, pred, label, node_to_index, feature_to_name):
        # update count_flows
        self.count_flows += 1
        
        # define colors
        src_ip, dst_ip = row[feature_to_name['Source IP']], row[feature_to_name['Destination IP']]
        # src_color, dst_color = self.get_color(src_ip), self.get_color(dst_ip)
        src_color, dst_color = "lightskyblue", "lightskyblue"
        
        # check label
        src_label = src_ip in [attacker_ip, victom_ip]
        dst_label = dst_ip in [attacker_ip, victom_ip]
        
        # add nodes
        src_port, dst_port = row[feature_to_name['Source Port']], row[feature_to_name['Destination Port']]
        src, dst = f'{src_ip}:{src_port}', f'{dst_ip}:{dst_port}'
        src_id, dst_id = self.get_id(src), self.get_id(dst)
        
        if dataset_type in ['labeled_data', 'elastic_flows']:
            src_label = dst_label = row[feature_to_name['Label']] == feature_to_name['Attack Label']
            src_ip = f'{src_ip}_{src_label}'
            src, dst = f'{src}_{src_label}', f'{dst}_{dst_label}'
            
        if not self.graph.has_node(src_ip):
            node_to_index[src_ip] = len(pred)
            pred.append(False)
            label.append(src_label)
            self.graph.add_node(src_ip, side = 'Client-IP', amount = 0, length = 0, time_delta = 0.0, 
                                min_packet_length = 0, max_packet_length = 0, mean_packet_length = 0,
                                FIN_count = 0,  SYN_count = 0,  RST_count = 0,  PSH_count = 0,  ACK_count = 0,  
                                URG_count = 0,
                                anomaly_score_history =  [], cluster = -1, printed = False,
                                pred = False, label = src_label, cluster_pred = False, ann_pred = False,
                                ip = src_ip, port = None, flows = 1, color = src_color)
                
        if not self.graph.has_node(src_id):
            node_to_index[src_id] = len(pred)
            pred.append(False)
            label.append(src_label)
            self.graph.add_node(src_id, side = 'Client', amount = 0, length = 0, time_delta = 0.0, 
                                min_packet_length = 0, max_packet_length = 0, mean_packet_length = 0,
                                FIN_count = 0,  SYN_count = 0,  RST_count = 0,  PSH_count = 0,  ACK_count = 0,  
                                URG_count = 0,
                                anomaly_score_history =  [], cluster = -1, printed = False,
                                pred = False, label = src_label, cluster_pred = False, ann_pred = False,
                                ip = src_ip, port = src_port, flows = 0, color = src_color)

        if not self.graph.has_node(dst_id):
            node_to_index[dst_id] = len(pred)
            pred.append(False)
            label.append(dst_label)
            self.graph.add_node(dst_id, side = 'Server', amount = 0, length = 0, time_delta = 0.0, 
                                min_packet_length = 0, max_packet_length = 0, mean_packet_length = 0,
                                FIN_count = 0,  SYN_count = 0,  RST_count = 0,  PSH_count = 0,  ACK_count = 0,  
                                URG_count = 0,
                                anomaly_score_history =  [], cluster = -1, printed = False,
                                pred = False, label = dst_label, cluster_pred = False, ann_pred = False,
                                ip = dst_ip, port = dst_port, sip = src, flows = 0, color = dst_color)
        
        # add edges
        if not self.graph.has_edge(src_ip, src_id):
            self.graph.add_edge(src_ip, src_id)
        if not self.graph.has_edge(src_id, dst_id):
            self.graph.add_edge(src_id, dst_id)
        
        # update nodes features
        self.update_features(src_id, row, 'Fwd', feature_to_name)
        self.update_features(src_ip, row, 'Fwd', feature_to_name)
        self.update_features(dst_id, row, 'Bwd', feature_to_name)
        
    def update_features(self, id, row, direction, feature_to_name):
        # Update features of a node in the graph
        node = self.graph.nodes[id]
        
        node["flows"] += 1
        node['amount'] += int(row[feature_to_name.get(f'amount_{direction}')])
        node['length'] += float(row[feature_to_name.get(f'length_{direction}')])
        node['min_packet_length'] += float(row[feature_to_name.get(f'min_packet_length_{direction}')])
        node['max_packet_length'] += float(row[feature_to_name.get(f'max_packet_length_{direction}')])
        node["packet_index"] = row[feature_to_name["Timestamp"]]
        
        if node['side'] == 'Client-IP':
            node["flows"] = self.graph.degree(id)
            
        if  node['amount'] != 0:     
            node['mean_packet_length'] = node['length']/node['amount']
            
        flags = {
                'FIN': int(row[feature_to_name['FIN']]),
                'SYN': int(row[feature_to_name['SYN']]),
                'RST': int(row[feature_to_name['RST']]),
                'PSH': int(row[feature_to_name['PSH']]),
                'ACK': int(row[feature_to_name['ACK']]),
                'URG': int(row[feature_to_name['URG']]),
            }
        
        for flag in flags:
            node[f'{flag}_count'] += flags[flag]             