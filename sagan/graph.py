import requests
from ripe.atlas.sagan import TracerouteResult
import numpy as np
import pyasn
import networkx as nx
import matplotlib.pyplot as plt
import time 
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from collections import defaultdict

asndb = pyasn.pyasn('ipasn_db.dat')
G = nx.Graph()

page_size = 500
stop = int(time.time()) # time now
start = stop - 86400 # time yesterday

url = f"https://atlas.ripe.net/api/v2/measurements/?type=traceroute&status=4&start_time__gte={start}&page_size={page_size}"
response = requests.get(url)
print(response.raise_for_status())
data = response.json()
res = [m['id'] for m in data['results']]
measurement_ids = np.array(res)
print(measurement_ids.size)
print(measurement_ids.shape)

arr_np = np.array(measurement_ids)
#print("Tried to found %d measurements, in the end, found %d measurements" % (n_measurement, arr_np.size))

for measurement_id in measurement_ids:
    source = 'https://atlas.ripe.net/api/v2/measurements/' + str(measurement_id)+ '/results/'
    response = requests.get(source)
    data = response.json()

    for x in data:
        parsed = TracerouteResult(x)
        path_ip = parsed.ip_path

        asn_paths = []

        for i, hop in enumerate(path_ip): # Iterate through the hops of a traceroute
            #filter IP None values
            hop = [x for x in hop if x is not None]
            hop = [x for x in hop if x != 'None']
            
            new_hop = [] # ASn(s) in the hop
            for j, ip in enumerate(hop):
                asn = asndb.lookup(ip)[0] # Get ASN for IP
                if asn is not None and asn != 'None': # ASN not a None value
                    if asn not in new_hop:
                        new_hop.append(asn) 
            
            n_asn = len(new_hop) # Number of ASns in the hop
            
            if n_asn > 1: # Save all paths and end here
                asn_paths.append(new_hop)
                break
            elif n_asn == 1:
                asn = new_hop[0]
                asn_paths.append(asn)

        path_size = len(asn_paths)

        if path_size > 0: # Non empty path
            last_elem = asn_paths[-1]
            
            if isinstance(last_elem, int) == False: # A hop contained multiple ASns
                if len(last_elem) > 1: 
                    print("A hop contained multiple ASns")
                    flat_path = [asn_paths[0]]
                    for i in range(1, path_size - 1): # Flatten the path --> No consecutive same ASNs in the path
                        x = asn_paths[i]
                        last_elem = flat_path[-1]
                        if x == last_elem:
                            continue
                        else:
                            flat_path.append(x)

                    flat_path = np.array(flat_path)
                    flat_path_size = flat_path.size

                    last_flat_path = flat_path[-1]

                    G.add_node(flat_path[0])
                    for i in range(1, flat_path_size):
                        prev = flat_path[i-1]
                        asn = flat_path[i]
                        G.add_node(asn)
                        G.add_edge(asn, prev)
                    for _, asn in last_elem:
                        G.add_node(asn)
                        G.add_edge(asn, last_flat_path)
            else: # Every hop had exactly one ASn
                flat_path = [asn_paths[0]]
                for i in range(1, path_size): # Flatten the path --> No consecutive same ASNs in the path
                    x = asn_paths[i]
                    last_elem = flat_path[-1]
                    if x == last_elem:
                        continue
                    else:
                        flat_path.append(x)

                flat_path = np.array(flat_path)
                flat_path_size = flat_path.size
                G.add_node(flat_path[0])
                for i in range(1, flat_path_size):
                    prev = flat_path[i-1]
                    asn = flat_path[i]
                    G.add_node(asn)
                    G.add_edge(asn, prev)

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()


# degree part
degrees = [deg for _, deg in G.degree()]
print(degrees)

#  histogram 
plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=range(1, max(degrees)+2), edgecolor='black', color='skyblue')
plt.title("Distribution of Node Degrees in ASN Graph")
plt.xlabel("Degree (Number of Connections)")
plt.ylabel("Number of ASNs")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



all_nodes = list(G.nodes()) # convert nodes to a list
num_nodes = len(all_nodes)
num_samples = 1000  #Total  Number of random node pairs to pick to find shortest path

# Score: how many times each node appears in the middle of shortest paths
score = defaultdict(int) # empty dictionary to keep count how many times the nodes apprears

for _ in range(num_samples):
    src, dst = random.sample(all_nodes, 2)# picking nodes
    try:
        path = nx.shortest_path(G, source=src, target=dst)#finding shortest path between the 2
        for node in path[1:-1]:  # skip first and last
            score[node] += 1
    except nx.NetworkXNoPath:
        continue  # no path between this pair, skip that pair and continues

used_nodes = len(score) #how many nodes were used
all_scores = list(score.values()) #list of all score 

print(f"\nTotal nodes in graph: {num_nodes}")
print(f"Nodes used in paths (score ≥ 1): {used_nodes}")

# top asn with high score help us know important asns or central asns
top = sorted(score.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 nodes by score:")
for node, val in top:
    print(f"Node {node} → {val}")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(all_scores, bins=30, edgecolor='black', color='skyblue')
plt.title("Node Path Score Histogram")
plt.xlabel("Times appeared in shortest paths")
plt.ylabel("Number of nodes")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

