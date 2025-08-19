import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime

# CSV dosyasını yükle
df = pd.read_csv("IRIS_deprem.csv")

# Saat biçimini datetime objesine çevir
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + ' ' + df['Time'])

# Zaman özelliklerini çıkar
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['second'] = df['datetime'].dt.second

# Özellik ve hedef değişkenleri ayır
X = df[['Year','Mag']] #  ,'Lat','Lon','Depth','Timestamp']]  # Girdiler
y = df[[ 'Month', 'Day']]  # Tahmin edilecek: tarih

# Ölçeklendirme
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Eğitim-test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


# modeli oluştur ve eğit
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128)],
    'max_iter': [500, 1000, 1500],
    'alpha': [0.0001, 0.001, 0.01],
}

grid_search = GridSearchCV(MLPRegressor(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

# # MLP modeli oluştur ve eğit
# model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
# model.fit(X_train, y_train)

# Tahmin yap
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Orijinal ölçeğe çevir

# Hata hesapla
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
r2 = r2_score(scaler_y.inverse_transform(y_test), y_pred)
print(r2)

# Kullanıcıdan büyüklük (Mag) ve yıl (Year) bilgilerini al
user_year = int(input("Tahmin yapmak istediğiniz yılı girin: "))  # Örnek: 2020
user_mag = float(input("Tahmin yapmak istediğiniz büyüklüğü girin: "))  # Örnek: 6.5

# Kullanıcı girdisini uygun formata sokalım
user_input = scaler_X.transform([[user_year, user_mag]])

# Model ile tahmin yapalım
predicted_scaled = model.predict(user_input)
predicted = scaler_y.inverse_transform(predicted_scaled)

# Tahmin edilen ay ve günü gösterelim
predicted_month = round(predicted[0][0])
predicted_day = round(predicted[0][1])

# Sonuçları yazdıralım
print(f"Tahmin edilen ay: {predicted_month}")
print(f"Tahmin edilen gün: {predicted_day}")

# ####
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from datetime import datetime

# # CSV dosyasını yükle
# df = pd.read_csv("senin_csv_dosyan.csv")

# # Saat biçimini datetime objesine çevir
# df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + ' ' + df['Time'])

# # Zaman özelliklerini çıkar
# df['hour'] = df['datetime'].dt.hour
# df['minute'] = df['datetime'].dt.minute
# df['second'] = df['datetime'].dt.second

# # Giriş (X) ve çıkış (y) tanımı
# features = ['Lat', 'Lon', 'Depth', 'Year', 'Month', 'Day', 'hour', 'minute', 'second']
# X = df[features]
# y = df['Mag']

# # Veriyi ölçeklendir
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Eğitim ve test verisine ayır
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # MLP modeli tanımla ve eğit
# model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
# model.fit(X_train, y_train)

# # Tahmin ve değerlendirme
# y_pred = model.predict(X_test)
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# ####