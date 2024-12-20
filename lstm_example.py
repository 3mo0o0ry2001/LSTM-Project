import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. إنشاء بيانات تسلسلية بسيطة
data = np.sin(np.linspace(0, 100, 200))  # دالة جيبية كنموذج للبيانات
data = data.reshape(-1, 1)

# 2. تطبيع البيانات بين 0 و 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# 3. تقسيم البيانات إلى بيانات تدريب واختبار
train_size = int(len(data_normalized) * 0.8)
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]

# 4. إنشاء تسلسلات (Features and Labels)
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# 5. بناء نموذج LSTM
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(sequence_length, 1)),
    Dense(1)  # طبقة إخراج لتوقع قيمة واحدة
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. تدريب النموذج
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# 7. التنبؤ بالبيانات
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # عكس التطبيع

# 8. عرض النتائج
import matplotlib.pyplot as plt

# الرسم البياني للبيانات الحقيقية والتوقعات
plt.plot(scaler.inverse_transform(test_data[sequence_length:]), label="Actual Data")
plt.plot(predictions, label="Predictions")
plt.legend()
plt.show()