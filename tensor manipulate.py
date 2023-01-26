import numpy as np
import torch
# Numpy로 1차원 텐서인 벡터 만들기
t=np.array([0.,1.,2.,3.,4.,5.,6.])
# 1차원 텐서인 벡터의 차원과 크기 출력
#ndim은 차원을 알려줌 shape는 크기를 출력 -> (1x7)의 크기를 가지고 있음
print('Rank of t: ', t.ndim)
print('shate of t:' ,t.shape)
# 2차원 행렬 2D with Numpy
#차원은 2차원 행렬 (4X3)행렬 4행 3열
t=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
print(t)
print('Rank of t: ', t.ndim)
print('shate of t:' ,t.shape)

#파이토치로 1차원 텐서인 벡터 만들기
#dim -> 차원 , shape와 size() 원소의 개수
t=torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t)
print(t.dim())
print(t.shape)
print(t.size())

#2D with pyTorch
#텐서의 차원 2차원 (4x3) 크기를 가짐
t = torch.FloatTensor([[1., 2., 3.],[4., 5., 6.],[7., 8., 9.],[10., 11., 12.]])
print(t)
print(t.dim())  # rank
print(t.size()) # shape

#브로드 캐스팅: 자동으로 크기를 맞춰서 연산을 수행하게 만듬
#크기가 같음
m1=torch.FloatTensor([[3,3]])
m2=torch.FloatTensor([[2,2]])
print(m1+m2)
#크기가 다름 Vector+scalar
m1=torch.FloatTensor([[1,2]])
m2=torch.FloatTensor([3]) # [3] -> [3,3] m2의크기를 [2,2]와 같이 변경
print(m1+m2)

#뷰- 원소의 수를 유지하면서 텐서의 크기 변경
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
# 3차원 텐서에서 2차원 텐서로 변경
#ft라는 텐서를 (?,3)의크기로 변경 -1-> 첫번째 차원은 잘 모르겠으니 파이토치에게 맡김
#변경전 텐서의 원소의 개수 (2x2x3)=12개 변경후 (4x3)=12개
print(ft.view([-1,3]))
print(ft.view([-1,3]).shape)
#3차원 텐서의 크기 변경 텐서의 크기를 변경하더라도 원소의 수는 유지 되어야 한다
print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)
