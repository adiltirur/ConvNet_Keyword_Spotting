import os
path = '/home/adil/NN_Project/dataset/renamed/ten'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.wav'))
    i = i+1
