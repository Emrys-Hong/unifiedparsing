import json
import os
import adeseg

with open('./object150_info.csv', 'r') as f:
    object150_info = f.readlines()[1:]
    object150_info = [x.strip().split(',') for x in object150_info]

with open('./label_assignment.csv', 'r') as f:
    label_assignment = f.readlines()[1:]
    label_assignment = [x.strip().split(',') for x in label_assignment]

with open('./object_part_hierarchy.csv', 'r') as f:
    object_part_hierarchy = f.readlines()[1:]
    object_part_hierarchy = [x.strip().split(',') for x in object_part_hierarchy]


# add object 150 labels into a set
valid_object_name = set()
for line in object150_info:
    valid_name = line[-1]
    valid_object_name.add(valid_name)

# generate new label assignment file

# 1) remove non-ade labels + ade scene labels
label_assignment = [x for x in label_assignment if x[0] == 'ade20k' and x[1] != 'scene']

# 2) remove object labels outside of 150 labels
new_label_assignment = []
valid_object_id = set()
for line in label_assignment:
    if line[1] == 'object' and line[3] not in valid_object_name:  # raw_name
        line[4] = '0'
        line[5] = '-'

    if line[4] != '0' and line[1] == 'object':
        valid_object_id.add(line[-2])  # broden_id
    new_label_assignment.append(line)

label_assignment = new_label_assignment.copy()
print(len(valid_object_id))

# 3) remove parts that do not belong to 150 labels
object_part_hierarchy = [x for x in object_part_hierarchy if x[0] in valid_object_id]

valid_part_id = set()
for line in object_part_hierarchy:
    parts = line[-2].split(';')
    for part in parts:
        valid_part_id.add(part)

new_label_assignment = []
for line in label_assignment:
    if line[1] == 'part' and line[-2] not in valid_part_id:
        line[4] = '0'
        line[5] = '-'
    new_label_assignment.append(line)
label_assignment = new_label_assignment.copy()

# 4) re-generate object id
object_name2new_id = dict()
for line in object150_info:
    object_name2new_id[line[-1]] = line[0]

new_label_assignment = []
for line in label_assignment:
    if line[1] == 'object' and line[3] in object_name2new_id:
        line[-2] = object_name2new_id[line[3]]
    new_label_assignment.append(line)
label_assignment = new_label_assignment

# 5) re-generate part id
part_old_id2new_id = {'0': '0'}
cnt = 0
new_label_assignment = []
for line in label_assignment:
    if line[1] == 'part' and line[4] != '0':
        old_part_id = line[4]
        if old_part_id not in part_old_id2new_id:
            cnt += 1
            part_old_id2new_id[old_part_id] = str(cnt)
        line[4] = part_old_id2new_id[old_part_id]
    new_label_assignment.append(line)
label_assignment = new_label_assignment


# 6) re-generate object-part hierarchy
new_object_part_hierarchy = []
for line in object_part_hierarchy:
    new_line = line[:-2]
    part_labels = line[-2].split(';')
    part_names = line[-1].split(';')

    part_labels = [part_old_id2new_id[x] for x in part_labels if x in part_old_id2new_id]
    part_names = [x for x, y in zip(part_names, part_labels) if y in part_old_id2new_id]
    new_line += [';'.join(part_labels), ';'.join(part_names)]
    new_object_part_hierarchy.append(new_line)

# 7) write out
with open('./new_label_assignment.csv', 'w') as f:
    f.write('dataset,category,raw_label,raw_name,broden_label,broden_name\n')
    for line in label_assignment:
        f.write(','.join(line)+'\n')

with open('./new_object_part_hierarchy.csv', 'w') as f:
    f.write('object_label,object_name,part_labels,part_names\n')
    for line in new_object_part_hierarchy:
        f.write(','.join(line)+'\n')

# 8) restore training and validation file list
ade = adeseg.AdeSegmentation(
                directory=os.path.join('./', "ade20k"),
                version='ADE20K_2016_07_26')
from IPython import embed
embed()

with open('./broden_ade20k_pascal_train.json', 'r') as f:
    filelist = f.readlines()

with open('./broden_ade20k_pascal_val.json', 'r') as f:
    filelist += f.readlines()

filelist = [json.loads(x) for x in filelist]
filelist = [x for x in filelist if x['dataset'] == 'ade20k']

train_set = set()
val_set = set()
with open('./ADE20K_object150_train.txt', 'r') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    for line in lines:
        train_id = int(line.split('_')[-1].split('.')[0])
        train_set.add(train_id)
with open('./ADE20K_object150_val.txt', 'r') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    for line in lines:
        val_id = int(line.split('_')[-1].split('.')[0])
        val_set.add(val_id)

train_list = []
val_list = []

for line in filelist:
    idx = line['file_index']
    if idx in train_set:
        assert idx not in val_set
        train_list.append(line)
    else:
        assert idx in val_set, line
        val_list.append(line)

with open('ade20k_train.json', 'w') as f:
    for line in train_list:
        del line['nr_part']
        f.write(json.dumps(line)+'\n')

with open('ade20k_val.json', 'w') as f:
    for line in val_list:
        del line['nr_part']
        f.write(json.dumps(line)+'\n')
