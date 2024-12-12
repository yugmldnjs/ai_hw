import numpy as np
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 불러옴
from sklearn import datasets     # 붓꽃 데이터를 가져오기 위해 불러옴
from sklearn.model_selection import train_test_split  # 훈련 샘플과 테스트 샘플을 나누기 위해 불러옴

# 붓꽃 데이터 로드
iris = datasets.load_iris()


# 시그모이드 함수
def actf(x):
    return 1 / (1 + np.exp(-x))


# 시그모이드 함수의 미분치, 시그모이드 함수 출력값을 입력으로 받는다.
def actf_deriv(out):
    return out * (1 - out)


# 입력유닛의 개수, 은닉유닛의 개수, 출력유닛의 개수
inputs, hiddens, outputs = 4, 5, 3
learning_rate = 0.01  # 학습률
epoch = 500            # 최대 학습 반복 횟수

# 데이터 샘플 및 답
X = iris.data
T = []
# 0,1,2로 구분되어 있는 답을 [1,0,0],[0,1,0],[0,0,1] 형식으로 변환
for i in iris.target:
    tmp = []
    for j in range(3):
        if i == j:
            tmp.append(1)
        else:
            tmp.append(0)
    T.append(tmp)

# 훈련 샘플, 테스트 샘플 분할
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, random_state=3)

# 초기 가중치 랜덤으로 초기화
W1 = np.round(np.random.rand(4, 5), 2)  # 소수점 셋째자리에서 반올림
W2 = np.round(np.random.rand(5, 3), 2)  # 소수점 셋째자리에서 반올림
B1 = np.round(np.random.rand(5), 1)  # 소수점 둘째자리에서 반올림
B2 = np.round(np.random.rand(3), 1)  # 소수점 둘째자리에서 반올림

# 에포크별 평균 손실 값 저장
train_losses = []
# 에포크별 정확도 저장
train_acc = []


# 순방향 전파 계산
def predict(x):
    layer0 = x  # 입력을 layer0에 대입한다.
    Z1 = np.dot(layer0, W1) + B1  # 행렬의 곱을 계산한다.
    layer1 = actf(Z1)  # 활성화 함수를 적용한다.
    Z2 = np.dot(layer1, W2) + B2  # 행렬의 곱을 계산한다.
    layer2 = actf(Z2)  # 활성화 함수를 적용한다.
    return layer0, layer1, layer2


# 역방향 전파 계산
def fit():
    global W1, W2, B1, B2  # 우리는 외부에 정의된 변수를 변경해야 한다.
    for i in range(epoch):
        total_loss = 0     # 손실 합 저장
        total_correct = 0  # 예측값 실제값 맞은 개수
        for x, y in zip(X_train, T_train):  # 학습 샘플을 하나씩 꺼낸다.
            x = np.reshape(x, (1, -1))  # 2차원 행렬로 만든다. ①
            y = np.reshape(y, (1, -1))  # 2차원 행렬로 만든다.

            layer0, layer1, layer2 = predict(x)  # 순방향 계산

            # 손실 함수 값 계산 (MSE)
            total_loss += MSE(layer2, y)

            # 학습 정확도 계산
            if accuracy(layer2, y):
                total_correct += 1

            layer2_error = layer2 - y  # 오차 계산
            layer2_delta = layer2_error * actf_deriv(layer2)  # 출력층의 델타 계산
            layer1_error = np.dot(layer2_delta, W2.T)  # 은닉층의 오차 계산 ②
            layer1_delta = layer1_error * actf_deriv(layer1)  # 은닉층의 델타 계산 ③

            W2 += -learning_rate * np.dot(layer1.T, layer2_delta)  # ④
            W1 += -learning_rate * np.dot(layer0.T, layer1_delta)  #
            B2 += -learning_rate * np.sum(layer2_delta, axis=0)  # ⑤
            B1 += -learning_rate * np.sum(layer1_delta, axis=0)  #

        train_losses.append(total_loss / len(T_train))  # 에포크별 평균 손실 함수 값
        train_acc.append(total_correct / len(T_train))  # 에포크별 정확도 추가


def test():
    total_loss = 0    # 총 손실 값 저장
    test_correct = 0  # 출력과 답이 맞는 경우의 수 저장
    for x, y in zip(X_test, T_test):
        x = np.reshape(x, (1, -1))  # 하나의 샘플을 꺼내서 2차원 행렬로 만든다.
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)  # 출력층의 값을 출력해본다.

        # 손실 함수 값 계산 (MSE)
        total_loss += MSE(layer2, y)

        if accuracy(layer2, y):  # 예측 값과 실제 답 확인
            test_correct += 1

    # 테스트 평균 손실 함수 값 계산
    test_loss = total_loss / len(T_test)

    # 테스트 정확도 계산
    test_accuracy = test_correct / len(T_test)
    return test_loss, test_accuracy


# MSE 계산
def MSE(pred, target):
    return np.round(0.5*np.sum((pred - target) ** 2), 5)


# 정확도 계산
def accuracy(pred, target):
    # 예측 값와 실제 값 추출
    predictions = np.argmax(pred)
    targets = np.argmax(target)

    # 비교한 bool값 반환
    return predictions == targets


# 에포크별 학습데이터 정확도 그래프 그리기
def train_accuracy_graph():
    plt.plot(train_acc, label="Accuracy")
    plt.title("Accuracy Progression by Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


# 손실함수 그래프 그리기
def loss_graph():
    plt.plot(train_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss Function Graph")
    plt.legend()
    plt.grid()

    plt.ylim(0, max(train_losses) * 1.1)  # y축 범위를 0에서 손실 최대값의 10% 여유까지 설정
    y_ticks = np.linspace(0, max(train_losses) * 1.1, num=10)  # y축 눈금을 10개로 나눔
    plt.yticks(y_ticks, [f"{tick:.4f}" for tick in y_ticks])  # 눈금의 값을 소수점 4자리로 표시

    plt.show()


# 가중치 출력
print("=" * 28, "가중치", "=" * 28)
print("초기 가중치")
print('-' * 50)
print(f"W1\n{W1}\nW2\n{W2}\nB1\n{B1}\nB2\n{B2}")
print('-' * 50)

# 학습
fit()
print("최종 가중치")
print('-' * 50)
print(f"W1\n{W1}\nW2\n{W2}\nB1\n{B1}\nB2\n{B2}")

# 테스트 결과 출력
print("=" * 28, "test 결과", "=" * 28)
test_loss, test_accuracy = test()
print('-' * 50)
print(f"epoch: {epoch}")
print(f"learning_rate: {learning_rate}")
print(f"test accuracy: {test_accuracy:.2f}")
# 에포크별 학습 정확도 그래프 출력
train_accuracy_graph()
print(f"train loss(last epoch): {train_losses[-1]}")
print(f"test loss: {test_loss}")
# 에포크별 평균 손실 그래프 출력
loss_graph()

