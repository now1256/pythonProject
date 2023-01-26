import torch
import numpy as np
# 텐서와 배여과 행렬과 매우 유사한 특수한 자료구조 pytorch에서는 텐서를 사용
#데이터로부터 직접 텐서 생성 가능
#데이터의 자료형은 자동으로 유추
data= [[1,2],[3,4]]

#데이터를 텐서로 변환
x_data=torch.tensor(data)

# numpy배열을 텐서로 변환 반대도 가능
x_np=torch.from_numpy(np.array(data))

#명시적으로 재정의 하지 않는다면 x_data의 속성(shape,datatype)을 유지
#f-string 과 \n 줄바꿈이 사용
x_ones=torch.ones_like(x_data)
# rand_like(): random한 float의 숫자 대입
x_rand=torch.rand_like(x_data,dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

#shape는 텐서의 차원을 나타내는 튜플로 아래 함수들에서는 출력 텐서의 차원을 결정
shape=(2,3,)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones([2,3,])
zeros_tensor=torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
#텐서의 속성
print(f"Shape of tensor: {x_data.shape}")
print(f"Datatype of tensor: {x_data.dtype}")
print(f"Device tensor is stored on: {x_data.device}")