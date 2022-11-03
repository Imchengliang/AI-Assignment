import matplotlib.pyplot as plt
import xlsxwriter

def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1, len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1, len(obj_list)+1)
    plt.show()

def plotRoutes(model):
    for route in model.best_sol.routes:
        x_coord=[model.depot_dict[route[0]].x_coord]
        y_coord=[model.depot_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0] == 'd1':
            plt.plot(x_coord, y_coord, marker='o', color='black', linewidth=0.5, markersize=5)
        elif route[0] == 'd2':
            plt.plot(x_coord, y_coord, marker='o', color='orange', linewidth=0.5, markersize=5)
        else:
            plt.plot(x_coord, y_coord, marker='o', color='b', linewidth=0.5, markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.show()

def outPut(model):
    work = xlsxwriter.Workbook('result4.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, 'opt_type')
    worksheet.write(1, 0, 'obj')
    worksheet.write(0, 1, 'drive distance of vehicles')

    worksheet.write(1, 1, model.best_sol.obj)
    for row, route in enumerate(model.best_sol.routes):
        worksheet.write(row+2, 0, 'v'+str(row+1))
        r = [str(i+1) for i in route]
        worksheet.write(row+2, 1, '-'.join(r))
    work.close()