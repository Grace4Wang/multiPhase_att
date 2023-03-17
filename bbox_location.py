from read_roi import read_roi

def write_location(roi_path, file_name,patient):
    '''
    :param roi_path:
    :param file_name:
    :param patient:
    :return:

    根据read—roi函数的结果来写location.txt
    '''
    file = open(str(file_name) + ".txt", 'a')
    location = read_roi(roi_path,patient)
    for i in location:
        file.write(str(i) + ' ')
    file.write('\n')
    file.close()


# 最大的bbox
def largest_bbox(file_name):
    '''

    根据bbox的txt为文件来计算所有样本中最大的肿瘤的bbox

    :param file_name:
    :return:
    '''
    x,y,z=[],[],[]
    with open(str(file_name)+'.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[0:-1]
            a = line.split(' ')
            x.append(int(a[7]))
            y.append(int(a[8]))
            z.append(int(a[9]))
        x.sort()
        y.sort()
        z.sort()
    return x[-1],y[-1],z[-1]
