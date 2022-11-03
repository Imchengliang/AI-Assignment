import matplotlib.pyplot as plt
import pandas as pd

def plotRoutes():
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'pink']
    i = 0
    for route in list4:
        x_coord=[]
        y_coord=[]
        for j in route:
            x_coord.append(j[0])
            y_coord.append(j[1])
        print(x_coord, y_coord)
        plt.grid()
        plt.plot(x_coord, y_coord, marker='o', color=colors[i], linewidth=0.5, markersize=5)
        i += 1
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__=='__main__':
    df = pd.read_excel('result1.xlsx')
    list1, list2, list3, list4 = [], [], [], []
    for i in range(1, 10):
        a = df.iloc[i, 1]
        s = a.split('-') 
        list2.append(list(filter(lambda x:x!='',s)))
        for j in range(len(list2[i-1])):
            b = int(list2[i-1][j])
            list2[i-1][j] = b
        list2[i-1].insert(0, 0)
        list2[i-1].append(0)

    df1 = pd.read_excel('2.xlsx')
    
    for i in list2:
        for j in i:
            list3.append([df1.iloc[[j],[2]].values[0][0], df1.iloc[[j],[3]].values[0][0]])
        list4.append(list3)
        list3 = []

    plotRoutes()
    