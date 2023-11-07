from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

def worker_func(item):
    # Выполнить свою работу внутри потока
    result = item * 2
    return result

def process_func(item):
    # Создать пул потоков
    with ThreadPool(2) as executor:
    # with ThreadPoolExecutor() as executor:
        # Применить функцию к каждому элементу списка с использованием потоков
        results = list(executor.map(worker_func, item))
    return results

if __name__ == '__main__':
    # Создание пула процессов
    pool = Pool(2)

    # Создание списка
    my_list = [[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]]

    # Применение функции к каждому элементу списка с использованием пула процессов
    results = pool.map(process_func, my_list)

    # Вывод результатов
    print(results)