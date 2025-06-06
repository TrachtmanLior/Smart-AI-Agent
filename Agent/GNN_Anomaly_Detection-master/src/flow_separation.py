from pyshark import FileCapture
import numpy as np

from vector import Vector
from tri_graph import TriGraph
from visualization import plot_ann_indexes
from network import ANN
from results import measure_results
from execute_pipeline import execute_pipeline

def update_flow_state(flow, packet):
    fin_flag = packet.tcp.flags_fin == '1'
    ack_flag = packet.tcp.flags_ack == '1'
    
    current_state = flow.state
    
    if current_state == 'ESTABLISHED':
        if fin_flag:
            flow.state = 'FIN_WAIT'
    
    elif current_state == 'FIN_WAIT':
        if ack_flag and fin_flag:
            flow.state = 'CLOSE_WAIT'
        elif ack_flag:
            flow.state = 'FIN_WAIT_ACKED'
    
    elif current_state == 'FIN_WAIT_ACKED':
        if fin_flag:
            flow.state = 'CLOSE_WAIT'
    
    elif current_state == 'CLOSE_WAIT':
        if ack_flag:
            flow.state = 'CLOSED'

    if packet.tcp.flags_reset == '1':
        flow.state = 'CLOSED'

def find_packet_time(packet):
    ts = int(float(packet.frame_info.time_epoch))
    return ts

def separate_packets_pcap(pcap_file, num_of_rows=-1, algo='ann', plot=True, num_of_flows=2000):
    
    if algo == 'network':
        ann = ANN()
    elif algo in ['ann', 'clustering', 'combined']:
        tri_graph = TriGraph()
    
    def flow_finished(vector):
        if algo == 'network' and ann.add_vector(vector)[0] == 'anomaly':
            print(f'anomaly on index {i}, stream: {stream_number}, vector: {vector}\n')
        elif algo in ['ann', 'clustering', 'combined']:
            tri_graph.add_separated_flow_to_graph(vector)
    
    cap = FileCapture(pcap_file)
    prev_count_flows = 0

    streams = {}
    
    for i, packet in enumerate(cap):
        if i == num_of_rows:
            break
        
        if i % 10000 == 0:
            print(f'processed {i} packets')
            
        # Check only TCP packets
        if hasattr(packet, 'ip') and hasattr(packet, 'tcp'):

            # Skip retransmission packets
            if 'analysis_retransmission' in dir(packet.tcp):
                continue

            # Get the ip, port for the src and dst
            src_ip, dst_ip = packet.ip.src, packet.ip.dst
            dest_port = packet[packet.transport_layer].dstport if hasattr(
                packet, 'transport_layer') else None
            src_port = packet[packet.transport_layer].srcport if hasattr(
                packet, 'transport_layer') else None
            
            # Get the stream numner from the TCP packet
            stream_number = int(packet.tcp.stream)

            flags = {
                'FIN': int(packet.tcp.flags_fin),
                'SYN': int(packet.tcp.flags_syn),
                'RST': int(packet.tcp.flags_reset),
                'PSH': int(packet.tcp.flags_push),
                'ACK': int(packet.tcp.flags_ack),
                'URG': int(packet.tcp.flags_urg),
            }
            
            if int(src_port) > int(dest_port):
                src, dst, fwd = f'{src_ip}:{src_port}', f'{dst_ip}:{dest_port}', True
            else:
                dst, src, fwd = f'{src_ip}:{src_port}', f'{dst_ip}:{dest_port}', False
            
            if stream_number not in streams: # Got a new flow number
                # Skip single resets packets
                if packet.tcp.flags_reset == '1':
                    continue
                
                streams[stream_number] = Vector(len(packet), src, dst, fwd, stream_number, flags)
            else: # New packet of existing flow
                vector = streams[stream_number]
                # Aggregate the packet's feature to the existing flow
                vector.add_packet(len(packet), packet.tcp.time_delta, src, flags)

            vector = streams[stream_number]
            update_flow_state(vector, packet)
            # End a flow in FYN or RST flag is opened
            if vector.state == 'CLOSED':
                # Add the whole flow - after he terminated to tri_graph
                vector.packet_index = find_packet_time(packet)
                flow_finished(vector)
                streams.pop(stream_number)
            
        if algo == 'network':
            if plot:
                plot_ann_indexes(np.array(ann.vectors))
            continue
        
        # Compute the embeddings and the ANN every X flows
        if tri_graph.count_flows - prev_count_flows >= num_of_flows:
            execute_pipeline(tri_graph, algo, plot)
            prev_count_flows = tri_graph.count_flows
        

    execute_pipeline(tri_graph, algo, plot)
    
    measure_results(tri_graph.graph)
