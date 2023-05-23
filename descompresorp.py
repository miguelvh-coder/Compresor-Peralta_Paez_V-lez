import sys
import time
import struct
import numpy as np
from mpi4py import MPI

def decompressp(file_path='comprimidop.elmejorprofesor'):
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        with open(file_path, 'rb') as f:
            data = f.read()
        compressed_data = np.frombuffer(data, dtype=np.uint32)
        dictionary = {i: bytes([i]) for i in range(256)}
    else:
        compressed_data = None
        dictionary = None

    dictionary = comm.bcast(dictionary, root=0)

    chunk_size = len(compressed_data) // size
    remainder = len(compressed_data) % size

    compressed_data_local = np.empty(chunk_size + 1, dtype=np.uint32)
    comm.Scatter(compressed_data, compressed_data_local, root=0)

    result = []
    w = bytes([compressed_data_local[0]])
    result.append(w)

    for k in compressed_data_local[1:]:
        if k in dictionary:
            entry = dictionary[k]
        else:
            entry = w + bytes([w[0]])
        result.append(entry)
        dictionary[len(dictionary)] = w + bytes([entry[0]])
        w = entry

    result = comm.gather(result, root=0)

    if rank == 0:
        result = np.concatenate(result)
        output_file_path = 'descomprimidop-elmejorprofesor.txt'
        with open(output_file_path, 'wb') as f:
            f.write(b"".join(result))

        end_time = time.time()
        print(end_time - start_time)

if __name__ == "__main__":
    file_path = sys.argv[1]
    decompressp(file_path)