import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import numpy as np
from adopt import ADOPT
### 1B
# def df_w(W):
#     """
#     Thực hiện tính gradient của dw1 và dw2
#     Arguments:
#     W -- np.array [w1, w2]
#     Returns:
#     dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
#     """
#     #################### YOUR CODE HERE ####################


#     dW = np.array([0.2 * W[0], 4 * W[1]])
#     ########################################################

#     return dW

# def sgd(W, dW, lr):
#     """
#     Thực hiện thuật tóa Gradient Descent để update w1 và w2
#     Arguments:
#     W -- np.array: [w1, w2]
#     dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
#     lr -- float: learning rate
#     Returns:
#     W -- np.array: [w1, w2] w1 và w2 sau khi đã update
#     """
#     #################### YOUR CODE HERE ####################


#     W = W - (lr * dW)
#     ########################################################
#     return W

# def train_p1(optimizer, lr, epochs):
#     """
#     Thực hiện tìm điểm minimum của function (1) dựa vào thuật toán
#     được truyền vào từ optimizer
#     Arguments:
#     optimize : function thực hiện thuật toán optimization cụ thể
#     lr -- float: learning rate
#     epoch -- int: số lượng lần (epoch) lặp để tìm điểm minimum
#     Returns:
#     results -- list: list các cặp điểm [w1, w2] sau mỗi epoch (mỗi lần cập nhật)
#     """

#     # initial point
#     W = np.array([-5, -2], dtype=np.float32)
#     # list of results
#     results = [W]
#     #################### YOUR CODE HERE ####################
#     # Tạo vòng lặp theo số lần epochs
#     # tìm gradient dW gồm dw1 và dw2
#     # dùng thuật toán optimization cập nhật w1 và w2
#     # append cặp [w1, w2] vào list results

#     for i in range(epochs):
#         dW = df_w(W)
#         W = optimizer(W, dW, lr)
#         results.append(W)
#     for i in results:
#         print(i)
#     ##################################
#     return results

## 2B
# def df_w(w):
#     """
#     Thực hiện tính gradient của dw1 và dw2
#     Arguments:
#     W -- np.array [w1, w2]
#     Returns:
#     dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
#     """
#     #################### YOUR CODE HERE ####################


#     dW = np.array([0.2 * w[0], 4 * w[1]])
#     ########################################################

#     return dW


# def sgd_momentum(W, dW, lr, V, beta):
#     """
#     Thực hiện thuật tóan Gradient Descent + Momentum để update w1 và w2
#     Arguments:
#     W -- np.array: [w1, w2]
#     dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
#     lr -- float: learning rate
#     V -- np.array: [v1, v2] Exponentially weighted averages gradients
#     beta -- float: hệ số long-range average
#     Returns:
#     W -- np.array: [w1, w2] w1 và w2 sau khi đã update
#     V -- np.array: [v1, v2] Exponentially weighted averages gradients sau khi đã cập nhật
#     """
#     #################### YOUR CODE HERE ####################

#     V = beta * V + (1 - beta) * dW
#     W = W - (lr * V)
#     ########################################################
#     return W, V

# def train_p1(optimizer, lr, epochs):
#     """
#     Thực hiện tìm điểm minimum của function (1) dựa vào thuật toán
#     được truyền vào từ optimizer
#     Arguments:
#     optimize : function thực hiện thuật toán optimization cụ thể
#     lr -- float: learning rate
#     epochs -- int: số lượng lần (epoch) lặp để tìm điểm minimum
#     Returns:
#     results -- list: list các cặp điểm [w1, w2] sau mỗi epoch (mỗi lần cập nhật)
#     """
#     # initial
#     W = np.array([-5, -2], dtype=np.float32)
#     V = np.array([0, 0], dtype=np.float32)
#     results = [W]
#     #################### YOUR CODE HERE ####################
#     # Tạo vòng lặp theo số lần epochs
#     # tìm gradient dW gồm dw1 và dw2
#     # dùng thuật toán optimization cập nhật w1, w2, v1, v2
#     # append cặp [w1, w2] vào list results
#     for i in range(epochs):
#         dW = df_w(W)
#         W, V = optimizer(W, dW, lr, V, 0.5)
#         results.append(W)
#     for i in results:
#         print(i)
#     ########################################################
#     return results
## 3B
# def df_w(w):
#     """
#     Thực hiện tính gradient của dw1 và dw2
#     Arguments:
#     W -- np.array [w1, w2]
#     Returns:
#     dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
#     """
#     #################### YOUR CODE HERE ####################


#     dW = np.array([0.2 * w[0], 4 * w[1]])
#     ########################################################

#     return dW

# def RMSProp(W, dW, lr, S, gamma):
#     """
#     Thực hiện thuật tóan RMSProp để update w1 và w2
#     Arguments:
#     W -- np.array: [w1, w2]
#     dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
#     lr -- float: learning rate
#     S -- np.array: [s1, s2] Exponentially weighted averages bình phương gradients
#     gamma -- float: hệ số long-range average
#     Returns:
#     W -- np.array: [w1, w2] w1 và w2 sau khi đã update
#     S -- np.array: [s1, s2] Exponentially weighted averages bình phương gradients sau khi đã cập nhật
#     """
#     epsilon = 1e-6
#     #################### YOUR CODE HERE ####################

#     S = gamma * S + (1 - gamma) * (dW ** 2)

#     W = W - (lr / np.sqrt(S + epsilon)) * dW
#     ########################################################
#     return W, S

# def train_p1(optimizer, lr, epochs):
#     """
#     Thực hiện tìm điểm minimum của function (1) dựa vào thuật toán
#     được truyền vào từ optimizer
#     Arguments:
#     optimize : function thực hiện thuật toán optimization cụ thể
#     lr -- float: learning rate
#     epochs -- int: số lượng lần (epoch) lặp để tìm điểm minimum
#     Returns:
#     results -- list: list các cặp điểm [w1, w2] sau mỗi epoch (mỗi lần cập nhật)
#     """
#     # initial
#     W = np.array([-5, -2], dtype=np.float32)
#     S = np.array([0, 0], dtype=np.float32)
#     results = [W]
#     #################### YOUR CODE HERE ####################
#     # Tạo vòng lặp theo số lần epochs
#     # tìm gradient dW gồm dw1 và dw2
#     # dùng thuật toán optimization cập nhật w1, w2, s1, s2
#     # append cặp [w1, w2] vào list results
#     for i in range(epochs):
#         dW = df_w(W)
#         W, S = optimizer(W, dW, lr, S, 0.9)
#         results.append(W)
#     for i in results:
#         print(i)
#     ########################################################
#     return results
## 4B
# def df_w(w):
#     """
#     Thực hiện tính gradient của dw1 và dw2
#     Arguments:
#     W -- np.array [w1, w2]
#     Returns:
#     dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
#     """
#     #################### YOUR CODE HERE ####################


#     dW = np.array([0.2 * w[0], 4 * w[1]])
#     ########################################################

#     return dW

# def Adam(W, dW, lr, V, S, beta1, beta2, t):
#     epsilon = 1e-6
#     V = beta1*V + (1-beta1)*dW
#     S = beta2*S + (1-beta2)*dW**2
#     V_coor = V/(1-beta1**t)
#     S_coor = S/(1-beta2**t)
#     W = W - lr*V_coor/(np.sqrt(S_coor) + epsilon)
#     return W, V, S

# def train_p1(optimizer, lr, epochs):
#     """
#     Thực hiện tìm điểm minimum của function (1) dựa vào thuật toán
#     được truyền vào từ optimizer
#     Arguments:
#     optimize : function thực hiện thuật toán optimization cụ thể
#     lr -- float: learning rate
#     epochs -- int: số lượng lần (epoch) lặp để tìm điểm minimum
#     Returns:
#     results -- list: list các cặp điểm [w1, w2] sau mỗi epoch (mỗi lần cập nhật)
#     """
#     # initial
#     W = np.array([-5, -2], dtype=np.float32)
#     V = np.array([0, 0], dtype=np.float32)
#     S = np.array([0, 0], dtype=np.float32)
#     results = [W]
#     #################### YOUR CODE HERE ####################
#     # Tạo vòng lặp theo số lần epochs
#     # tìm gradient dW gồm dw1 và dw2
#     # dùng thuật toán optimization cập nhật w1, w2, s1, s2, v1, v2
#     # append cặp [w1, w2] vào list results
#     # các bạn lưu ý mỗi lần lặp nhớ lấy t (lần thứ t lặp) và t bất đầu bằng 1
#     for i in range(epochs):
#         dW = df_w(W)
#         W, V, S = optimizer(W, dW, lr, V, S, 0.9, 0.999, i + 1)
#         results.append(W)

#     for i in results:
#         print(i)

#     ########################################################
#     return results
## 5
class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, hidden_dims)
        self.layer4 = nn.Linear(hidden_dims, hidden_dims)
        self.layer5 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        x = self.layer5(x)
        x = self.sigmoid(x)
        out = self.output(x)
        return out
if __name__ == '__main__':
    # train_p1(sgd, lr=0.4, epochs=30)
    # train_p1(sgd_momentum, lr=0.6, epochs=30)
    # train_p1(RMSProp, lr=0.3, epochs=30)
    # train_p1(Adam, lr=0.2, epochs=30)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    batch_size = 512
    num_epochs = 300
    lr = 0.01

    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size)

    model = MLP(input_dims=784, hidden_dims=128, output_dims=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # SGD
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # SGD + Momentum
#    optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=0.9)
#    RMSProp
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    # Adam
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # ADOPT
    # optimizer = ADOPT(model.parameters(), lr=lr)
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()
        t_loss = 0
        t_acc = 0
        cnt = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_acc += (torch.argmax(outputs, 1) == y).sum().item()
            cnt += len(y)
        t_loss /= len(train_loader)
        train_losses.append(t_loss)
        t_acc /= cnt
        train_acc.append(t_acc)

        model.eval()
        v_loss = 0
        v_acc = 0
        cnt = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                v_loss += loss.item()
                v_acc += (torch.argmax(outputs, 1)==y).sum().item()
                cnt += len(y)
        v_loss /= len(test_loader)
        val_losses.append(v_loss)
        v_acc /= cnt
        val_acc.append(v_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Train_Loss: {t_loss:.4f}, Train_Acc: {t_acc:.4f}, Validation Loss: {v_loss:.4f}, Val_Acc: {v_acc:.4f}")