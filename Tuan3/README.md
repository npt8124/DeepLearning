BTVN 1
Giải thích data_pd[-1]
ở trường hợp 1
import numpy as np
import pandas as pd

numpy_arr = np.arange(5)
data_pd = pd.Series(numpy_arr)
data_pd[-1]
Series này có index mặc định: 0,1,2,3,4
Khi viết data_pd[-1], pandas hiểu rằng lấy phần tử có label = -1
Nhưng trong index không có -1 -> KeyError

trường hợp 2:
data_pd = pd.Series([1.25,2.0,0.75,1.0], index=['a','b','c','d'])
data_pd[-1]
Lúc này index là 'a', 'b', 'c', 'd'
Không có label -1, nên Pandas fallback sang positional indexing, hiểu là lấy phần tử cuối cùng
Nên kết quả: 1.0