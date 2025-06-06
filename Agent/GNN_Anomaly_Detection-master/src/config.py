from feature_to_name import feature_to_name_CIC_2017, feature_to_name_IoT, feature_to_name_elastic

features = [ 'amount', 
            'length', 
            'mean_packet_length', 
            'FIN_count', 
            'SYN_count', 
            'RST_count', 
            'PSH_count', 
            'ACK_count', 
            'URG_count',
            # 'count_opened_sockets', 
            # 'min_packet_length', 
            # 'max_packet_length',
            # 'time_delta'
            ]

anomaly_score_history_size  = 10
ann_threshold = 15
ann_history_threshold = 20
clustering_threshold = 5
network_threshold = 14

hidden_size = 128
output_size = 64

attacker_ip = "172.16.0.1"
victom_ip = "192.168.10.50"

dataset_type = 'csv'

feature_to_name = feature_to_name_CIC_2017