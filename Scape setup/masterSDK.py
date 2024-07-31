import sys
import os
import time
import msvcrt
import MultiCast
import Capture

sys.path.append(r'C:\Users\Student\Documents\ASJO\SDK\MultiCast')
sys.path.append(r'C:\Users\Student\Documents\ASJO\SDK\AinstecCamSDK_Python\samples')

def count_py_files(directory):
    py_file_count = 0

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.py'):
                py_file_count += 1

    return py_file_count

if __name__ == "__main__":
    directory_path = r'C:\Users\Student\Documents\ASJO\SDK\data\multicast\camera1'
    image_count = count_py_files(directory_path)

    MultiCast.main(image_count)  
    time.sleep(1)
    Capture.main(image_count)
