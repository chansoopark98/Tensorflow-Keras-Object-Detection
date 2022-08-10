import glob

# 
label_path = '/home/park/park/Tensorflow-Keras-Object-Detection/data_generate/data/coex_hand/labels/train/00000.txt'
with open(label_path, "r") as file:
    read_bbox_list = file.readlines()
    # print(list(read_bbox_list))
    for i in range(len(read_bbox_list)):
        line_label = read_bbox_list[i].replace('\n', '')
        line_label = list(line_label.split(' '))
        print()

        label = int(line_label[0])
        bbox = line_label[1:5]
        bbox = [float(box) for box in bbox]
        