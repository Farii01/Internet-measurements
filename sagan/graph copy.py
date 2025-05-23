import requests
from ripe.atlas.sagan import TracerouteResult
import numpy as np
import pyasn
import networkx as nx
import matplotlib.pyplot as plt

asndb = pyasn.pyasn('ipasn_db.dat')
G = nx.Graph()

# # 100274119, 103112978
# measurement_ids = [100274119, 103112978, 103112523]

n_measurement = 100

rnd_ids = np.random.randint(1, 103160683, size=n_measurement)
measurement_ids = []
# gap_limit = ??
# rnd_ids = [1,-2,-2]

for id in rnd_ids:
    url = f"https://atlas.ripe.net/api/v2/measurements/?type=traceroute&status=4&id={id}&af=4&is_public=true"
    response = requests.get(url)
    try:
        response.raise_for_status()
        data = response.json()
        res = [m['id'] for m in data['results']]
        if res:
            val = res[0]
            if isinstance(val, int):
                measurement_ids.append(res[0])
    except:
        print("error, lets move on")
        continue

print(measurement_ids)
arr_np = np.array(measurement_ids)
print("Tried to found %d measurements, in the end, found %d measurements" % (n_measurement, arr_np.size))


for measurement_id in measurement_ids:
    #measurement_id = 24733372
    print("found ID: ",measurement_id)
    source = 'https://atlas.ripe.net/api/v2/measurements/' + str(measurement_id)+ '/results/'
    response = requests.get(source)
    data = response.json()

    for x in data:
        parsed = TracerouteResult(x)
        path_ip = parsed.ip_path

        print(path_ip)

        arr = np.array(path_ip)

        #print(arr.shape[1])

        paths = []
        for i in range(1, arr.shape[1] + 1):
            paths.append(arr[:,i-1])

        # path_1 = arr[:,0]
        # path_2 = arr[:,1]
        # path_3 = arr[:,2]

        for path in paths:
            path = np.array(path)
            #print("1. before:", path)
            path = path[path != (None)] # remove nones
            #print("1. after:", path)
            for idx, x in enumerate(path):
                asn = asndb.lookup(x)[0]
                path[idx] = asn
            
            #print("2. before:", path)
            path = np.array(list(filter(lambda x: x != None, path))) # Remove NONE values
            path = np.array(list(filter(lambda x: x != 'None', path))) # Remove NONE values
            #print("2. after:", path)
            path_size = path.size

            if path_size > 0:
                flat_path = [path[0]]
                for i in range(1, path_size - 1):
                    x = path[i]
                    last_elem = flat_path[-1]
                    if x == last_elem:
                        continue
                    else:
                        flat_path.append(x)

                flat_path = np.array(flat_path) 
                # print("original path", path)
                # print("flat path", flat_path)

                flat_path_size = flat_path.size
                G.add_node(flat_path[0])
                for i in range(1, flat_path_size):
                    prev = flat_path[i-1]
                    asn = flat_path[i]
                    G.add_node(asn)
                    G.add_edge(asn, prev)


nx.draw(G, with_labels=True, font_weight='bold')
plt.show()
