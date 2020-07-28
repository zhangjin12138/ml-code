import numpy as np 

def check(x,y,w,b): # 检查是否正确分类
    wx = w * x
    wx = sum(wx)
    result = y * (wx + b)
    if result > 0:
        return True
    else:
        return False

def regirt(x,y,n,w,b):# 将w， b 进行更新
    w = w + n*x*y
    b = b + n*y
    return w, b

def find_all(x_data, y_data, w, b): # 感知机流程
    data_len = len(x_data)
    i = 0
    while(i<data_len):
        print('正在测试',x_data[i], y_data[i])
        if check(x_data[i], y_data[i], w, b) == True:
            print(x_data[i], y_data[i], '分类正确')
            i += 1
        else:
            print( sum((w * x_data[i] + b) * y_data[i]), '< 0 ')
            w, b = regirt(x_data[i], y_data[i], n, w, b)
            print( 'w b 更新为:',w,b);
            i = 0
        print("\n")
    print('最终的w b 为:', w, b)
    return w, b 


if __name__ == "__main__":
    x_data = np.array([[3,3],[4,3],[1,1]])
    y_data = np.array([1,1,-1])
    w = 0
    b = 0
    n = 1
    find_all(x_data,y_data, w, b)