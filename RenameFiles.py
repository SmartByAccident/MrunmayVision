import os
path = "M:\\Mrunmay\\FTC 2025\\Vision\\Images\\Tests2\\"
file_list = os.listdir(path)
print(file_list)
file_list.sort()

for i, file in enumerate(file_list, start=1):
    new_name = "test" + str(i).zfill(2) + os.path.splitext(file)[1]
    os.rename(path + file, path + new_name)
    print(i)

