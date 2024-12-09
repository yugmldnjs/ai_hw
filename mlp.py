import numpy as np

# 시그모이드 함수
def actf(x):
	return 1/(1+np.exp(-x))

# 시그모이드 함수의 미분치, 시그모이드 함수 출력값을 입력으로 받는다.
def actf_deriv(out):
	return out*(1-out)

# 입력유닛의 개수, 은닉유닛의 개수, 출력유닛의 개수
inputs, hiddens, outputs = 2, 2, 1
learning_rate=0.2

# 훈련 샘플과 정답
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])

W1 = np.array([[0.10,0.20], [0.30,0.40]])
W2 = np.array([[0.50],[0.60]])
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])

# 순방향 전파 계산
def predict(x):
    layer0 = x			# 입력을 layer0에 대입한다.
    Z1 = np.dot(layer0, W1)+B1	# 행렬의 곱을 계산한다.
    layer1 = actf(Z1)		# 활성화 함수를 적용한다.
    Z2 = np.dot(layer1, W2)+B2	# 행렬의 곱을 계산한다.
    layer2 = actf(Z2)		# 활성화 함수를 적용한다.
    return layer0, layer1, layer2
# 역방향 전파 계산
def fit():
    global W1, W2, B1, B2		# 우리는 외부에 정의된 변수를 변경해야 한다.
    for i in range(90000):		# 9만번 반복한다.

        for x, y in zip(X, T):		# 학습 샘플을 하나씩 꺼낸다.
            x = np.reshape(x, (1, -1))	# 2차원 행렬로 만든다. ①
            y = np.reshape(y, (1, -1))	# 2차원 행렬로 만든다.

            layer0, layer1, layer2 = predict(x)			# 순방향 계산
            layer2_error = layer2-y				# 오차 계산
            layer2_delta = layer2_error*actf_deriv(layer2)	# 출력층의 델타 계산
            layer1_error = np.dot(layer2_delta, W2.T)		# 은닉층의 오차 계산 ②
            layer1_delta = layer1_error*actf_deriv(layer1)	# 은닉층의 델타 계산 ③

            W2 += -learning_rate*np.dot(layer1.T, layer2_delta)	# ④
            W1 += -learning_rate*np.dot(layer0.T, layer1_delta)	#
            B2 += -learning_rate*np.sum(layer2_delta, axis=0)	# ⑤
            B1 += -learning_rate*np.sum(layer1_delta, axis=0)	#
def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1))	# 하나의 샘플을 꺼내서 2차원 행렬로 만든다.
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)		# 출력층의 값을 출력해본다.
fit()
test()