# BTVN thú vị
# cho ma trận = [[99,99,99],
#              [99,99,99],
#              [99,99,99]]
# hãy tưởng tượng ô 99 trống
# giả sử 1 là x
# giả sử 0 là o
# Lấy đầu vào từ phía x và phía 0 luân phiên
# Nếu đầu vào từ phía x là (0,0)
# ma trận sẽ trở thành
# [[x,99,99],
#  [99,99,99],
#  [99,99,99]]
# Nếu đầu vào từ phía o cũng là (0,0)
# Thông báo họ nhập lại
# Nếu không thì điền vào ma trận
# Thử thách: Nếu người đầu tiên có 3 phần tử liên tiếp thì dừng trò chơi.
CODE:
board = [[99 for _ in range(3)] for _ in range(3)]

def print_board():
    for row in board:
        print(["X" if x == 1 else "O" if x == 0 else "_" for x in row])
    print()

def check_win(player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)):
        return True
    if all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

current_player = 1

print("Game Caro 3x3 bắt đầu!")
print("Nhập dạng: row col (ví dụ: 1 2)")
print_board()

while True:
    player_char = "X" if current_player == 1 else "O"
    raw = input(f"Lượt {player_char}: ").strip()

    # Kiểm tra input trống
    if raw == "":
        print("Không được để trống. Nhập lại!")
        continue

    parts = raw.split()
    if len(parts) != 2:
        print("Phải nhập đúng 2 số: row col")
        continue

    try:
        row, col = map(int, parts)
    except ValueError:
        print("Chỉ được nhập số nguyên!")
        continue

    if not (0 <= row <= 2 and 0 <= col <= 2):
        print("Chỉ được nhập số trong khoảng 0–2!")
        continue

    if board[row][col] != 99:
        print("Ô đã được đánh, nhập lại!")
        continue

    board[row][col] = current_player
    print_board()

    if check_win(current_player):
        print(f"Người chơi {player_char} thắng!")
        break

    current_player = 0 if current_player == 1 else 1
Công nghệ sử dụng: Python 3
Cách thức hoạt động: 
- Khởi tạo bàn cờ 3×3 với giá trị 99 (ô trống)
- Hai người chơi lần lượt nhập vị trí (row, col)
- 1 → X, 0 → O
- Không cho nhập sai định dạng, ngoài phạm vi hoặc ghi đè ô đã đánh
- Sau mỗi lượt, kiểm tra thắng theo hàng, cột, đường chéo
Kết quả: 
- Hiển thị bàn cờ sau mỗi lượt
- Trò chơi kết thúc khi có người thắng
- Thông báo người chiến thắng rõ ràng

# BTVN 2
y = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
# lấy 4,5,6
print(y[1])
# lấy 2,5
print(y[0][1], y[1][1])
# lấy 3,4
print([y[0][2], y[1][0]])
# lấy 9,6,3
print([y[2][2], y[1][2], y[0][2]])
Công nghệ sử dụng: Python 3
Cách thức hoạt động:
- Sử dụng list 2 chiều để lưu ma trận số
- Truy xuất phần tử bằng chỉ số hàng và cột theo cú pháp y[row][col]
Kết quả:
In đúng các giá trị cần lấy
[4, 5, 6]
2 5
[3, 4]
[9, 6, 3]

# BTVN 3
# cho x = [1,2,3,4,5,6,7,8,9,10]
# 1. Xuất ra các giá trị chẵn sử dụng if
# 2. Xuất ra các giá trị chẵn sử dụng list comprehension

x = [1,2,3,4,5,6,7,8,9,10]
# 1. Xuất ra các giá trị chẵn sử dụng if
for i in x:
    if i % 2 == 0:
        print(i)

# 2. Xuất ra các giá trị chẵn sử dụng list comprehension
even_numbers = [i for i in x if i % 2 == 0]
print(even_numbers)
Công nghệ sử dụng: Python 3
Cách thức hoạt động:
- Duyệt danh sách x bằng vòng lặp for, dùng điều kiện if i % 2 == 0 để kiểm tra số chẵn và in ra
- Sử dụng list comprehension để tạo nhanh danh sách mới chỉ chứa các số chẵn
Kết quả:
2
4
6
8
10
[2, 4, 6, 8, 10]

# BTVN
# Tạo một mảng numpy có kích thước 150x5. Giả sử mảng này chứa 150 mẫu dữ liệu của sinh viên bao gồm chiều cao, cân nặng, tuổi, lương và điểm trung bình
# Tách mảng dựa trên 4 cột đầu tiên thành một biến có tên là X và cột cuối cùng thành y
# Tách X thành X_train và X_test trong đó X_train chứa 70% dữ liệu và tách y thành Y_train và Y_test, trong đó Y_train chứa 70% dữ liệu
# Tạo 10 tập dữ liệu X_train không chồng chéo nhau

import numpy as np
from sklearn.model_selection import train_test_split

data = np.random.rand(150, 5) #Tạo mảng numpy 150x5

# Tách X và y
X = data[:, :4]   # 4 cột đầu
y = data[:, 4]    # cột cuối

# Chia train/test (70% train)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Tạo 10 tập dữ liệu X_train không chồng chéo nhau
splits = np.array_split(X_train, 10)

# Kiểm tra kích thước từng tập
for i, part in enumerate(splits):
    print(f"Fold {i+1}: shape = {part.shape}")

Công nghệ sử dụng: 
- Python 3
- NumPy
- scikit-learn (train_test_split)
Cách thức hoạt động:
- Tạo mảng NumPy kích thước 150×5 chứa dữ liệu sinh viên giả lập
- Tách 4 cột đầu làm X, cột cuối làm y
- Chia dữ liệu thành tập huấn luyện (70%) và kiểm tra (30%)
- Tách X_train thành 10 tập con không chồng chéo bằng np.array_split
Kết quả: Thu được:
X_train: ~105 mẫu
X_test: ~45 mẫu
10 tập con từ X_train, mỗi tập khoảng 10–11 mẫu, sẵn sàng cho cross-validation hoặc huấn luyện nhiều mô hình