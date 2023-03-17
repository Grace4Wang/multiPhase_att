'''
根据bbox的txt来计算当前肿瘤的bbox中心坐标
'''
x,y,z=[],[],[]
with open('bbox_location.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line[0:-1]
        a = line.split(' ')
        b = a[0].split('.')
        x_ = float(a[7])
        y_ = float(a[8])
        z_ = float(a[9])

        x.append(x_)
        y.append(y_)
        z.append(z_)
    x.sort()
    y.sort()
    z.sort()
    mid=int(0.5*len(x))+1
    print(z[mid],x[mid],y[mid])

