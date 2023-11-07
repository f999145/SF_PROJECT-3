import pandas as pd
import argparse
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from itertools import chain

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели  
from sklearn import metrics # инструменты для оценки точности модели  

def separation_list(input_list: list, sep: int=3) -> list[list[str]]:
    """
        Функция для дробления листа, что бы потом паралельно обрабатывать их паралельно.
        sep: Сколько элементов'.
    """
    out_list = list()
    # chunk_size = input_df.shape[0] // int(100/sep)
    for index in range(0, len(input_list), sep):
        out_list.append(input_list[index:index+sep])
    return out_list    

def separation_list2(input_list: list, sep: int=3) -> list[list[str]]:
    out_list = list()
    for index in range(0, len(input_list)):
        out_list.append(input_list[0:index+1])
    out_list = separation_list(out_list, sep)
    return out_list    



def _worker(in_pool: tuple[pd.DataFrame, str]):
    df, col = in_pool
    tmp = df.drop(columns=col, errors='ignore').copy()
    
    X = tmp.drop(['reviewer_score'], axis = 1)  
    y = tmp['reviewer_score'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    regr = RandomForestRegressor(n_estimators=100)  
    regr.fit(X_train, y_train)  
    y_pred = regr.predict(X_test)
    
    return (col, metrics.mean_absolute_percentage_error(y_test, y_pred))

def _process_worker(in_pool: tuple[pd.DataFrame, list[str]]):
    df, col = in_pool
    tmp_df = df.copy()
    len_col = len(col)
    with ThreadPool(processes=len_col) as thread:
        in_list = zip([tmp_df]*len_col, col)
        results = list(thread.map(_worker, in_list))
    return results

def main():
    # Создание парсера аргументов
    parser = argparse.ArgumentParser(description='Описание вашего скрипта')

    # Добавление аргумента
    parser.add_argument('-c', '--corr_limit', type=float, default=0.6, help='Верхняя граница лимита корреляции, то что ниже остается, то что выше удаляется')
    parser.add_argument('-cpu', '--cpu', type=int, default=4, help='Количество задействованных ядер процессора')
    parser.add_argument('-s', '--separation', type=int, default=2, help='Количество потоков для каждого процесса')
    parser.add_argument('-pl', '--proportion_low', type=float, default=0.01, help='Нижняя граница лимита булевых категорий (sum(True)). То что выше остается, ниже - удаляется.')
    parser.add_argument('-ph', '--proportion_hight', type=float, default=1, help='Верхняя граница лимита булевых категорий (sum(True)). То что выше удаляется, ниже - остается.')

    # Разбор аргументов командной строки
    args = parser.parse_args()


    df = pd.read_parquet('data/hotels_v3_parquet.gzip')
    cols_bool = pd.read_parquet('data/cols_bool_parquet.gzip')
    predict_col_mape = pd.read_parquet('data/fin_df_208_parquet.gzip').sort_values('MAPE', ignore_index=True)['col_name'].to_list()
    
    df['review_date_from_2015-01-01'] = (df['review_date'] - pd.to_datetime('2015-01-01')).dt.days.astype('UInt16')
    df = df.drop(columns=['hotel_address', 'negative_review', 'tags', 'positive_review', 'review_date'])
    
    for col in df.select_dtypes('category').columns:
        df[col] = (df[col].cat.codes + 1).fillna(0)

    df = df.fillna(0)

    corr_mat = df.corr().stack().reset_index(name="correlation")
    corr_mat['correlation_abs'] = corr_mat['correlation'].abs()

    
    cols = corr_mat.query(f'(correlation_abs > {args.corr_limit}) & (level_0 != level_1)')[['level_0', 'level_1']].values

    a_list = []
    b_list = []
    
    for a, b in cols:
        b_list.append(b)
        if not a in b_list:
            a_list.append(a)
    
    df = df.drop(columns=a_list, errors='ignore')
    cols_bool = cols_bool.reset_index(names='col_name')
    cols_bool = cols_bool[~cols_bool['col_name'].isin(a_list)].reset_index(drop=True)
    cols_bool_del = cols_bool.query(f'(proportion < {args.proportion_low} | proportion > {args.proportion_hight})')['col_name'].to_list()
    cols_bool = cols_bool.query(f'~(proportion < {args.proportion_low} | proportion > {args.proportion_hight})')['col_name'].to_list()
    
    
    df = df.drop(columns=cols_bool_del, errors='ignore')
    print('DataFrame:', df.shape)
    
    print('Start', datetime.strftime(datetime.now(timezone('Europe/Moscow')), '%Y-%b-%d %H:%M:%S'))
    
    with Pool(processes=args.cpu) as p:
        input_list = separation_list2(predict_col_mape, args.separation)
        input_list_len = len(input_list)
        pool_list = zip([df]*input_list_len, input_list)
        return_list = list(tqdm(p.imap(_process_worker, pool_list), total=input_list_len))
    
    return_list = chain.from_iterable(return_list)
    
    
    fin_df = pd.DataFrame(return_list, columns=['col_name', 'MAPE'])
    fin_df.to_parquet('data/fin_df_parquet.gzip', engine='pyarrow', compression='gzip')
    
    
    print('End', datetime.strftime(datetime.now(timezone('Europe/Moscow')), '%Y-%b-%d %H:%M:%S'))
    
if __name__ == '__main__':
    main()