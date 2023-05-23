import sys
import time
import struct
import numpy as np
from mpi4py import MPI

def compressp(file_path):
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        with open(file_path, 'rb') as f:
            data = f.read()
        file_size = len(data)
    else:
        data = None
        file_size = None

    file_size = comm.bcast(file_size, root=0)

    chunk_size = file_size // size
    remainder = file_size % size

    if rank == 0:
        for i in range(1, size):
            start = i * chunk_size
            end = start + chunk_size
            if i == size - 1:
                end = file_size
            chunk = data[start:end]
            comm.Send(chunk, dest=i)
        start = 0
        end = chunk_size
        if size == 1:
            end = file_size
        chunk = data[start:end]
        chunks = [chunk]
    else:
        chunks = None

    chunks = comm.bcast(chunks, root=0)

    dictionary = {bytes([i]): i for i in range(256)}
    result = []
    w = b""

    for chunk in chunks:
        for c in chunk:
            wc = w + bytes([c])
            try:
                if wc in dictionary:
                    w = wc
                else:
                    result.append(dictionary[w])
                    dictionary[wc] = len(dictionary)
                    w = bytes([c])
            except KeyError:
                pass
        if w:
            result.append(dictionary[w])

    compressed_data = comm.gather(result, root=0)

    if rank == 0:
        compressed_data = np.concatenate(compressed_data)
        compressed_data = np.array(compressed_data, dtype=np.uint32)

        compressed_bytes = struct.pack(f'<{len(compressed_data)}I', *compressed_data)
        with open('comprimidop.elmejorprofesor', 'wb') as f:
            f.write(compressed_bytes)

        end_time = time.time()
        print(end_time - start_time)

if __name__ == "__main__":
    file_path = sys.argv[1]
    compressp(file_path)