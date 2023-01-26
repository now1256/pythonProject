import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

X_train=torch.FloatTensor([[1],[2],[3]])
Y_train=torch.FloatTensor([[2],[4],[6]])

# 가중치
W = torch.zeros(1,requires_grad=True)
# 편향
b = torch.zeros(1, requires_grad=True)


# 경사하강법
opimizer= optim.SGD([W, b], lr=0.01)

nb_epochs=1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs+1):
    # 가설
    hypothesis = X_train * W + b
    # 비용 함수 선언
    cost = torch.mean((hypothesis - Y_train) ** 2)
    # cost로 가설을 개선
    opimizer.zero_grad() #초기화
    cost.backward()     #cost 미분
    opimizer.step()     # w,b값을 갱신 
    if epoch%100==0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))