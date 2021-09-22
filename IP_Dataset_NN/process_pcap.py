import dpkt
import datetime
from dpkt.utils import mac_to_str, inet_to_str
import numpy as np
import argparse
import sys

dtype = [('x','i4', 4),('y',np.float32)]

class Alist:
    def __init__(self):
        """First item of shape is ingnored, the rest defines the shape"""
        self.data = np.zeros(100,dtype=dtype)
        self.capacity = 100
        self.size = 0

    def update(self,x,y):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros(self.capacity,dtype=dtype)
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size]['x'] = x
        self.data[self.size]['y'] = y
        self.size += 1

    def finalize(self):
        return self.data[:self.size]


def process_packets(pcap, offset):
    """Print out information about each packet in a pcap
       Args:
           pcap: dpkt pcap reader object (dpkt.pcap.Reader)
    """
    
    arr = Alist()
    counts = {}
    window = [None] * offset

    i = 0
    # For each packet in the pcap process the contents
    for timestamp, buf in pcap:
        if i % 100000 == 0:
            print(i / 100000)
        # skip ipv6
        try:
            ip = dpkt.ip.IP(buf)
        except (dpkt.dpkt.UnpackError):
            continue

        # Update our results for index i-offset

        if (i >= offset):
            dst = window[i % offset]
            counts[dst] -= 1
            c = counts[dst]
            if c == 0:
                counts.pop(dst)
            dst_arr = list(map(int,dst.split('.')))
            arr.update(np.array(dst_arr), c)

        k = inet_to_str(ip.dst)

        if (k in counts):
            counts[k] += 1
        else:
            counts[k] = 1
        window[i%offset] = k
        
        i += 1

    return arr.finalize()

def count_max_packets(pcap, offset, m):
    """Print out information about each packet in a pcap
       Args:
           pcap: dpkt pcap reader object (dpkt.pcap.Reader)
    """
    
    arr = Alist()
    counts = {}
    window = [None] * offset
    bcounts = {}

    i = 0
    # For each packet in the pcap process the contents
    for timestamp, buf in pcap:

        # skip ipv6
        try:
            ip = dpkt.ip.IP(buf)
        except (dpkt.dpkt.UnpackError):
            continue

        # Update our results for index i-offset

        k = inet_to_str(ip.dst)
        src = inet_to_str(ip.src)

        if (src == m):
            if (i >= 2*offset):
                dst = window[i % offset]
                counts[dst] -= 1
                c = counts[dst]
                if c == 0:
                    counts.pop(dst)
                dst_arr = list(map(int,dst.split('.')))
                arr.update(np.array(dst_arr), c)
            if (i >= offset and i <= )
            if (i < offset)
            if (k in counts):
                counts[k] += 1
            else:
                counts[k] = 1
            window[i%offset] = k
            i += 1
            if i % 100000 == 0 and i > 0:
                print(i / 100000)

    return arr.finalize()

def find_max_packets(pcap):
    """Print out information about each packet in a pcap
       Args:
           pcap: dpkt pcap reader object (dpkt.pcap.Reader)
    """
    
    counts = {}

    i = 0
    # For each packet in the pcap process the contents
    for timestamp, buf in pcap:
        if i % 100000 == 0:
            print(i / 100000)
        # skip ipv6
        try:
            ip = dpkt.ip.IP(buf)
        except (dpkt.dpkt.UnpackError):
            continue
        k = inet_to_str(ip.src)

        if (k in counts):
            counts[k] += 1
        else:
            counts[k] = 1
        
        i += 1
    m = max(counts, key=counts.get)

    print(m)
    print(counts[m])
    return m

#output array for all destination IPs
def get_counts(fname, offset):
    """Open up a test pcap file and print out the packets"""
    counts = {}

    with open(fname, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        result = process_packets(pcap, offset)
    return result

#output array for all destination IPs from the heaviest source IP
def get_max_counts(fname, offset):
    """Open up a test pcap file and print out the packets"""
    counts = {}

    with open(fname, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        m = find_max_packets(pcap)

    with open(fname, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        result = count_max_packets(pcap, offset, m)

    return result



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--file", type=str, help="pcap file", default="")
    argparser.add_argument("--offset", type=int, help="How many too look into future", default=1000)
    argparser.add_argument("--heavy_src", type=int, help="False: consider all sources; True: consider only heaviest source", default=True)
    args = argparser.parse_args()

    if heavy_src:
        data = get_max_counts(args.file, args.offset)
        src = "_src"
    else:
        data = get_counts(args.file, args.offset)
        src = ""

    fname_arr = args.file.split('.')
    newfile = fname_arr[fname_arr.index('UTC')-1]

    #print out first 15 
    for i in range(15):
        print (data['y'][i])
    


    np.save("/home/tian/data/preprocessed/"+newfile+if_src+"_data", data)
