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
        # Distribute data to all processes
        for i in range(1, size):
            start = i * chunk_size
            end = start + chunk_size
            if i == size - 1:
                end = file_size  # Use the remaining length for the last chunk
            chunk = data[start:end]
            comm.Send(chunk, dest=i)
        # Process the first chunk
        start = 0
        end = chunk_size
        if size == 1:
            end = file_size  # Use the remaining length if only one process
        chunk = data[start:end]
        chunks = [chunk]  # Store the chunks in a list
    else:
        # Receive data chunks from process 0
        chunks = None

    # Broadcast the list of chunks to all processes
    chunks = comm.bcast(chunks, root=0)

    # Process each chunk in parallel
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