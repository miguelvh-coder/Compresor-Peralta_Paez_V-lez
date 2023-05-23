import sys
import filecmp

def verify(archivo1, archivo2):
    if filecmp.cmp(archivo1, archivo2):
        print("ok")
    else:
        print("nok")

if __name__ == "__main__":
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    verify(file_path_1, file_path_2)