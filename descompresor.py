import sys
import time
import numpy as np

def decompress(file_path='comprimido.elmejorprofesor'):
    start_time = time.time()

    with open(file_path, 'rb') as f:
        data = f.read()

    compressed_data = np.frombuffer(data, dtype=np.uint32)
    dictionary = {i: bytes([i]) for i in range(256)}
    result = []

    w = bytes([compressed_data[0]])
    result.append(w)

    for k in compressed_data[1:]:
        if k in dictionary:
            entry = dictionary[k]
        else:
            entry = w + bytes([w[0]])
        result.append(entry)
        dictionary[len(dictionary)] = w + bytes([entry[0]])
        w = entry

    output_file_path = 'descomprimido-elmejorprofesor.txt'
    with open(output_file_path, 'wb') as f:
        f.write(b"".join(result))

    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    file_path = sys.argv[1]
    decompress(file_path)