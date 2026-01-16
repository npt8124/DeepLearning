# BTVN 1:
#Cho y = 5x^5 + 6x^3 - 3x + 1. Cho biết độ dốc của đa thức trên ở điểm nào? 
- Code:
import torch

x = torch.tensor(2.0, requires_grad=True)

y = 5*x**5 + 6*x**3 - 3*x + 1
y.backward()

print("Giá trị y:", y.item())
print("Độ dốc tại x = 2:", x.grad.item())

Công nghệ sử dụng: python, pytorch
Cách thức hoạt động:
- Khởi tạo tensor x với requires_grad=True để PyTorch theo dõi phép toán
- Định nghĩa hàm số y từ x
- Gọi y.backward() để PyTorch tự động tính đạo hàm dy/dx
- Gradient được lưu trong x.grad
Kết quả: Giá trị y: 203.0
Độ dốc tại x = 2: 469.0

# BTVN 2:
Tạo một tensor ban đầu có giá trị là 2
Định nghĩa hàm số và tính gradient
y = x^3 2x^2 + 5x + 1
hãy tính dy/dx tại giá trị của x
Dùng phương pháp Gradient Descent với Learning rate alpha = 0.1 để cập nhật giá trị x trong 10 vòng lặp
- Code:
import torch

# Khởi tạo tensor
x = torch.tensor(2.0, requires_grad=True)

alpha = 0.1

for i in range(10):
    # Định nghĩa hàm
    y = x**3 + 2*x**2 + 5*x + 1
    # Tính gradient
    y.backward()
    print(f"Vòng {i+1}: x = {x.item():.6f}, dy/dx = {x.grad.item():.6f}")
    # Cập nhật x theo Gradient Descent
    with torch.no_grad():
        x -= alpha * x.grad
    # Reset gradient
    x.grad.zero_()
Công nghệ sử dụng: python, pytorch
Cách thức hoạt động:
- Khởi tạo tensor x = 2.0 với requires_grad=True để PyTorch theo dõi gradient
- Ở mỗi vòng lặp:
+ Tính giá trị hàm y
+ Gọi y.backward() để tự động tính đạo hàm dy/dx
+ Cập nhật x=x-a.(dy/dx) với a - 0.1
+ Reset gradient bằng x.grad.zero_() để tránh cộng dồng gradient
- Lặp lại 10 lần để quan sát sự thay đổi của x và gradient
Kết quả:
Vòng 1: x = 2.000000, dy/dx = 25.000000
Vòng 2: x = -0.500000, dy/dx = 3.750000
Vòng 3: x = -0.875000, dy/dx = 3.796875
Vòng 4: x = -1.254688, dy/dx = 4.703972
Vòng 5: x = -1.725085, dy/dx = 7.027413
Vòng 6: x = -2.427826, dy/dx = 12.971715
Vòng 7: x = -3.724998, dy/dx = 31.726830
Vòng 8: x = -6.897680, dy/dx = 120.143257
Vòng 9: x = -18.912006, dy/dx = 1002.343933
Vòng 10: x = -119.146408, dy/dx = 42116.011719

# BTVN 3:
Tạo một tập dữ liệu giả lập với x là số giờ học (ngẫu nhiên từ 1 - 10) và y là số điểm được tính theo công thức y = 3x + 5 + noise
Với noise là một giá trị ngẫu nhiên nhỏ
1. Khởi tạo tham số w và b ngẫu nhiên với requires_grad = True
2. Tính MSE
3. Tính Gradient
4. Cập nhật tham số w và b bằng Gradient Descent với Learning rate alpha = 0.01
5. Lặp lại quá trình trên trong 100 vòng lặp để xem sự hội tụ của mô hình
- Code:
import torch

# Tạo dữ liệu giả lập
torch.manual_seed(0)
N = 50
x = torch.rand(N, 1) * 9 + 1      # [1, 10]
noise = torch.randn(N, 1) * 0.5
y = 3 * x + 5 + noise

# 1. Khởi tạo tham số w và b
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

alpha = 0.01

for epoch in range(100):
    y_pred = w * x + b
    # 2. Tính MSE
    mse = torch.mean((y_pred - y) ** 2)
    # 3. Tính Gradient
    mse.backward()
    # 4. Cập nhật tham số
    with torch.no_grad():
        w -= alpha * w.grad
        b -= alpha * b.grad
    # Reset gradient
    w.grad.zero_()
    b.grad.zero_()
    # 5. Theo dõi hội tụ
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: MSE={mse.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")
Công nghệ sử dụng: python, pytorch
Cách thức hoạt động:
- Tạo dữ liệu giả lập
- Khởi tạo tham số w và b
- Tính MSE
- Gọi mse.backward() để pytorch tự động tính
- Cập nhật tham số (Gradient Descent)
- Reset gradient dùng w.grad.zero_() và b.grad.gradient_() để tránh cộng dồng gradient giữa các vòng lặp
- Lặp lại 100 epoch để theo dỗi sự hội tụ của mô hình
Kết quả:
Epoch  10: MSE=3.8913, w=3.7183, b=0.5902
Epoch  20: MSE=3.6232, w=3.6927, b=0.7514
Epoch  30: MSE=3.3751, w=3.6680, b=0.9065
Epoch  40: MSE=3.1456, w=3.6443, b=1.0557
Epoch  50: MSE=2.9331, w=3.6214, b=1.1992
Epoch  60: MSE=2.7366, w=3.5995, b=1.3373
Epoch  70: MSE=2.5547, w=3.5783, b=1.4701
Epoch  80: MSE=2.3864, w=3.5580, b=1.5978
Epoch  90: MSE=2.2306, w=3.5384, b=1.7207
Epoch 100: MSE=2.0865, w=3.5196, b=1.8389

# BTVN 5:
- Code:
import torch
x = torch.empty(3, 4)
print("empty:", x)
x = torch.zeros(2, 3)
print("zeros:", x)
x = torch.ones(2, 3)
print("ones:", x)
x = torch.rand(2, 3)      
print("random:", x)
x = torch.arange(12)
y = x.view(3, 4)
print("view:", y)
a = torch.zeros(2, 3)
b = torch.arange(6)
c = b.view_as(a)
print("view as:", c)

Công nghệ sử dụng: python, pytorch
Cách thức hoạt động:
1. torch.empty()
x = torch.empty(3, 4)
Cấp phát bộ nhớ nhưng không khởi tạo giá trị -> giá trị rác.
Kết quả: empty: tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])

2. torch.zeros()
x = torch.zeros(2, 3)
Tạo tensor toàn số 0
Kết quả: zeros: tensor([[0., 0., 0.],
        [0., 0., 0.]])

3. torch.ones()
x = torch.ones(2, 3)
Tạo tensor toàn số 1
Kết quả
ones: tensor([[1., 1., 1.],
        [1., 1., 1.]])

4. torch.rand()
x = torch.rand(2, 3)
Sinh số ngẫu nhiên phân phối đều trong khoảng (0, 1)
Kết quả: random: tensor([[0.1452, 0.8610, 0.8928],
        [0.0825, 0.1377, 0.6278]])
    
5. view()
x = torch.arange(12)
y = x.view(3, 4)
Thay đổi shape tensor mà không copy dữ liệu
Tensor mới dùng chung bộ nhớ với tensor cũ
Kết quả: tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

6. view_as()
a = torch.zeros(2, 3)
b = torch.arange(6)
c = b.view_as(a)
Reshape tensor b theo shape của tensor a
Kết quả: view as: tensor([[0, 1, 2],
        [3, 4, 5]])