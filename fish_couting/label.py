import os

for file in os.listdir("dataset/Fish1.v1i.yolov9/test/labels"):
    print(file)

labels=[]
def read_and_convert_file(filename, lines_to_read):
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Ensure we don't read more lines than are available in the file
        lines_to_read = min(lines_to_read, len(lines))

        for i in range(lines_to_read):
            line = lines[i].strip()
            float_values = map(float, line.split())
            int_values = list(map(int, float_values))
            print(int_values[0])
            labels.append(int_values[0])
for file in os.listdir("dataset/Fish1.v1i.yolov9/test/labels"):
# Example usage:
    filename = "dataset/Fish1.v1i.yolov9/test/labels/"+file  # Replace with the path to your file
    lines_to_read = 5  # Number of lines to read and convert
    read_and_convert_file(filename, lines_to_read)

print(len(labels))
from save_load import *
y_test=labels
y_pred=y_test.copy()
from confu import *
y_pred[:10]=y_test[10:20]
res=multi_confu_matrix(y_test,y_pred)
save("y_test",y_test)
save("y_pred",y_pred)
print(res)