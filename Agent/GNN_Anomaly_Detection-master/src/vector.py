from datetime import datetime

# Define a class to represent a vector of network traffic data
class Vector():
    def __init__(self, length, src, dst, fwd, stream_number, flags, packet_index = 0) -> None:
        # others
        self.src = src
        self.dst = dst
        self.stream_number = stream_number
        self.packet_index = packet_index
        self.state = 'ESTABLISHED'
        self.finished = False
        self.timestamp = datetime.fromtimestamp(packet_index).strftime('%Y-%m-%d %H:%M:%S')
        
        # features
        self.flags = flags
        self.fwd_packets_length = length if fwd else 0
        self.bwd_packets_length = 0 if fwd else length
        self.fwd_packets_amount = 1 if fwd else 0
        self.bwd_packets_amount = 0 if fwd else 1
        self.min_bwd_packet = length if fwd else 0
        self.min_fwd_packet = 0 if fwd else length
        self.max_bwd_packet = 0
        self.max_fwd_packet = 0
        self.time_delta = 0.0
        
    # Aggregate the features on existing stream
    def add_packet(self, length, time_delta, src, flags):
        if self.finished:
            self.reset()

        # Update features by direction
        if src == self.src:
            self.fwd_packets_amount += 1
            self.fwd_packets_length += length
            if self.min_fwd_packet == 0 or length < self.min_fwd_packet:
                self.min_fwd_packet = length
            if length > self.max_fwd_packet:
                self.max_fwd_packet = length
        else:
            self.bwd_packets_amount += 1
            self.bwd_packets_length += length
            if self.min_bwd_packet == 0 or length < self.min_bwd_packet:
                self.min_bwd_packet = length
            if length > self.max_bwd_packet:
                self.max_bwd_packet = length
            
        # Append time delta
        new_time_delta = self.time_delta + float(time_delta)
        self.time_delta = round(new_time_delta, 3)
        
        # Increase each flag
        for flag, value in flags.items():
            self.flags[flag] += int(value)

    def __str__(self) -> str:
        # Get all attribute names and values using vars()
        result = ''
        attributes = vars(self)
        for attr_name, attr_value in attributes.items():
            result += f'{attr_name}: {attr_value}, '
        return result

    # Reset a flow that has been terminated
    def reset(self):
        self.fwd_packets_length = 0
        self.bwd_packets_length = 0
        self.fwd_packets_amount = 0
        self.bwd_packets_amount = 0
        self.min_bwd_packet = 0
        self.min_fwd_packet = 0
        self.max_bwd_packet = 0
        self.max_fwd_packet = 0
        self.time_delta = 0.0
        self.packet_index = 0
        self.flags = {
            'FIN': 0,
            'SYN': 0,
            'RST': 0,
            'PSH': 0,
            'ACK': 0,
            'URG': 0,
        }
        self.state = 'ESTABLISHED'
        self.finished = False