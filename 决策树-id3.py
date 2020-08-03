import numpy as np
import pandas as pd
from collections import Counter


class Node(object):
    def __init__(self, x=None, label=None, y=None, data=None):
        self.label = label
        self.x = x
        self.y = y
        self.data = data
        self.child = []

    def append(self, node):
        self.child.append(node)

    def predict(self, features):
        if self.y is not None:
            return self.y
        for c in self.child:
            if c.x == features[self.label]:
                return c.predict(features)


class DTreeID3(object):
    def __init__(self, epsilon=0, alpha=0):
        # 信息增益阈值
        self.epsilon = epsilon
        self.alpha = alpha
        self.tree = Node()

    # 计算某列特征中每个种类的概率 
    # 输入为某一列
    # 输出为字典形式：特征的属性的名字和所占的百分比{'是': 0.625, '否': 0.375}
    def prob(self, datasets):
        datasets = pd.Series(datasets)
        data_len = len(datasets)
        p = {}
        vc = datasets.value_counts().values.tolist()
        vc_index = datasets.value_counts().index.tolist()
        for i in range(len(vc_index)):
            p[vc_index[i]] = vc[i] / data_len
        return p

    # 计算某一列的信息熵  
    def calc_ent(self, datasets):
        p = self.prob(datasets)
        value = list(p.values())
        sums = 0
        for i in value:
            sums += -i * np.log2(i)
        #print("sums:",sums) 
        #print("ss",-np.sum(np.multiply(value, np.log2(value))))
        #return -np.sum(np.multiply(value, np.log2(value)))
        return sums
    
    # 计算某列的条件熵
    def cond_ent(self, datasets, col):
        redata = datasets.T
        labelx = redata.columns.tolist()[col]
        labelx = redata[labelx].value_counts().index.tolist()
        p = {}
        for i in labelx:
            p[i] = redata.loc[redata[redata.columns[col]]==i][redata.columns[-1]].tolist()
        sums = 0
        for k in p.keys():
            sums += self.prob(datasets.iloc[col])[k] * self.calc_ent(p[k])
        return sums

    # 计算信息增益
    def info_gain_train(self, datasets, datalabels):
        datasets = datasets.T
        ent = self.calc_ent(datasets.iloc[-1])
        gainmax = {}
        for i in range(len(datasets) - 1):
            cond = self.cond_ent(datasets, i)
            gainmax[ent - cond] = i
        m = max(gainmax.keys())
        return gainmax[m], m

    # 训练
    def train(self, datasets, node):
        labely = datasets.columns[-1]
        # 判断样本是否为同一类输出Di，如果是则返回单节点树T。标记类别为Di
        if len(datasets[labely].value_counts()) == 1:
            node.data = datasets[labely]
            node.y = datasets[labely][0]
            return
        # 判断特征是否为空，如果是则返回单节点树T，标记类别为样本中输出类别D实例数最多的类别
        if len(datasets.columns[:-1]) == 0:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return
        gainmaxi, gainmax = self.info_gain_train(datasets, datasets.columns)
        if gainmax <= self.epsilon:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return
        vc = datasets[datasets.columns[gainmaxi]].value_counts()
        for Di in vc.index:
            node.label = gainmaxi
            child = Node(Di)
            node.append(child)
            new_datasets = pd.DataFrame([list(i) for i in datasets.values if i[gainmaxi]==Di], columns=datasets.columns)
            self.train(new_datasets, child)
            
    def fit(self, datasets):
        self.train(datasets, self.tree)
    
    def findleaf(self, node, leaf):
        for t in node.child:
            if t.y is not None:
                leaf.append(t.data)
            else:
                for c in node.child:
                    self.findleaf(c, leaf)

    def findfather(self, node, errormin):
        if node.label is not None:
            cy = [c.y for c in node.child]
            if None not in cy:  # 全是叶节点
                childdata = []
                for c in node.child:
                    for d in list(c.data):
                        childdata.append(d)
                childcounter = Counter(childdata)

                old_child = node.child  # 剪枝前先拷贝一下
                old_label = node.label
                old_y = node.y
                old_data = node.data

                node.label = None  # 剪枝
                node.y = childcounter.most_common(1)[0][0]
                node.data = childdata

                error = self.c_error()
                if error <= errormin:  # 剪枝前后损失比较
                    errormin = error
                    return 1
                else:
                    node.child = old_child  # 剪枝效果不好，则复原
                    node.label = old_label
                    node.y = old_y
                    node.data = old_data
            else:
                re = 0
                i = 0
                while i < len(node.child):
                    if_re = self.findfather(node.child[i], errormin)  # 若剪过枝，则其父节点要重新检测
                    if if_re == 1:
                        re = 1
                    elif if_re == 2:
                        i -= 1
                    i += 1
                if re:
                    return 2
        return 0

    def c_error(self):  # 求C(T)
        leaf = []
        self.findleaf(self.tree, leaf)
        leafnum = [len(l) for l in leaf]
        ent = [self.calc_ent(l) for l in leaf]
        error = self.alpha*len(leafnum)
        for l, e in zip(leafnum, ent):
            error += l*e
        return error

    def cut(self, alpha=0):  # 剪枝
        if alpha:
            self.alpha = alpha
        errormin = self.c_error()
        self.findfather(self.tree, errormin)
        
        
if __name__ == "__main__":

    def printnode(node, depth=0):  # 打印树所有节点
        if node.label is None:
            print(depth, (node.label, node.x, node.y, len(node.data)))
        else:
            print(depth, (node.label, node.x))
            for c in node.child:
                printnode(c, depth+1)
    datasets = np.array([
                   ['青年', '否', '否', '一般', '否'],
                   ['青年', '否', '否', '好', '否'],
                   ['青年', '是', '否', '好', '是'],
                   ['青年', '是', '是', '一般', '是'],
                   ['青年', '否', '否', '一般', '否'],
                   ['中年', '否', '否', '一般', '否'],
                   ['中年', '否', '否', '好', '否'],
                   ['中年', '是', '是', '好', '是'],
                   ['中年', '否', '是', '非常好', '是'],
                   ['中年', '否', '是', '非常好', '是'],
                   ['老年', '否', '是', '非常好', '是'],
                   ['老年', '否', '是', '好', '是'],
                   ['老年', '是', '否', '好', '是'],
                   ['老年', '是', '否', '非常好', '是'],
                   ['老年', '否', '否', '一般', '否'],
                   ['青年', '否', '否', '一般', '是']])  # 在李航原始数据上多加了最后这行数据，以便体现剪枝效果

    datalabels = np.array(['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
    train_data = pd.DataFrame(datasets, columns=datalabels)
    test_data = ['老年', '否', '否', '一般']

    dt = DTreeID3(epsilon=0)  # 可修改epsilon查看预剪枝效果
    dt.fit(train_data)
    
    y = dt.tree.predict(test_data)
    print('result:', y)

    dt.cut(alpha=0.5)  # 可修改正则化参数alpha查看后剪枝效果

    y = dt.tree.predict(test_data)
    print('result:', y)
