import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

# 파일 경로
train_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\범죄 예측\train.csv"
test_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\범죄 예측\test.csv"
sample_submission_file_path =  r"C:\Users\mytoo\OneDrive\바탕 화면\범죄 예측\sample_submission.csv"
submission_output_path = r"C:\Users\mytoo\OneDrive\바탕 화면\범죄 예측\my_submission.csv"

# 데이터 로드
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)
sample_submission = pd.read_csv(sample_submission_file_path)

train.info()

# 데이터를 확인
train.head(5)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
train['요일'].hist(bins=50, ax=axes[0])
axes[0].set_title('Histogram')
train['요일'].plot(kind='box', ax=axes[1])
axes[1].set_title('Boxplot')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
train['범죄발생지'].hist(bins=50, ax=axes[0])
axes[0].set_title('Histogram')
train['범죄발생지'].plot(kind='box', ax=axes[1])
axes[1].set_title('Boxplot')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
train['TARGET'].hist(bins=50, ax=axes[0])
axes[0].set_title('Histogram')
train['TARGET'].plot(kind='box', ax=axes[1])
axes[1].set_title('Boxplot')
plt.tight_layout()
plt.show()

# 불필요한 열 제거
if 'ID' in train.columns:
    train = train.drop('ID', axis=1)
if 'ID' in test.columns:
    test = test.drop('ID', axis=1)

# 순서형 특성 인코딩
ordinal_features = ['요일', '범죄발생지']

def vectorize_sequences(sequences, dimension=1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

for feature in ordinal_features:
    unique_values = list(train[feature].unique())
    train[feature] = train[feature].apply(lambda x: unique_values.index(x))
    test[feature] = test[feature].apply(lambda x: unique_values.index(x) if x in unique_values else len(unique_values))

# 타겟 변수 분리
y_train = train['TARGET']
X_train = train.drop('TARGET', axis=1)

# to_one_hot 함수 정의
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# 레이블 인코딩 -> 원-핫 인코딩으로 변경
y_train_one_hot = to_one_hot(y_train, dimension=y_train.nunique())

# 데이터 표준화
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (test - mean) / std

# 학습 데이터와 검증 데이터 분리
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=42)

# MLP 모델 설계
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_train_one_hot.shape[1], activation='softmax')
])
# 모델 컴파일
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train_split,
                    y_train_split,
                    epochs=30,
                    batch_size=50,
                    validation_data=(X_val_split, y_val_split))

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_train_one_hot.shape[1], activation='softmax')
])

del.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_split,
                    y_train_split,
                    epochs=15,
                    batch_size=50,
                    validation_data=(X_val_split, y_val_split))

val_loss, val_accuracy = model.evaluate(X_val_split, y_val_split)

# 예측 수행
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

# 제출 파일 생성
submission = pd.read_csv(sample_submission_file_path)
submission['TARGET'] = predicted_classes
submission.to_csv(submission_output_path, index=False)