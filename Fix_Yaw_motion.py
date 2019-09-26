import pandas as pd
import glob
from statistics import mean
import math

def recalculateYaw():
    reslist = glob.glob('*.res')
    id_list = []
    max_yaw_list = []
    min_yaw_list = []
    mean_yaw_list = []
    for res in reslist:
        id = res.split('\\')[-1].split('.')[0]
        id_list.append(id)
        try:
            res_df = pd.read_csv(res, index_col=0, sep= ',')
        except:
            res_df = pd.read_csv(res, index_col=0, sep=';')
        old_yaw_series_first = res_df['Body1_yaw_rad'].values[:3600*8-1]
        old_yaw_series_second = res_df['Body1_yaw_rad'].values[3600*8:]
        old_yaw_mean_first = mean(old_yaw_series_first)
        if old_yaw_mean_first >0:
            wrong_index = old_yaw_series_second < 0
            right_yaw = old_yaw_series_second[wrong_index] + math.pi
        elif old_yaw_mean_first <= 0:
            wrong_index = old_yaw_series_second >=0
            right_yaw = old_yaw_series_second[wrong_index] - math.pi

        old_yaw_series_second[wrong_index] = right_yaw
        new_yaw_series = old_yaw_series_second
        max_yaw_list.append(max(new_yaw_series))
        min_yaw_list.append(min(new_yaw_series))
        mean_yaw_list.append(mean(new_yaw_series))


        yaw_dict = {'max_yaw': max_yaw_list, 'min_yaw': min_yaw_list, 'mean_yaw': mean_yaw_list, 'index': id_list}
        with pd.ExcelWriter('New_yaw.xls') as writer:
            yaw_dict.to_excel(writer)
        writer.save()
        writer.close()

if __name__ == '__main__':
   recalculateYaw()