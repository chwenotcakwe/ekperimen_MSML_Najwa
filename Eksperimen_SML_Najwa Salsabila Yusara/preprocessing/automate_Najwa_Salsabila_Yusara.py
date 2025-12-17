import pandas as pd
from sklearn.model_selection import train_test_split

def clean_and_split_data():
    """
    Fungsi ini melakukan proses preprocessing otomatis:
    1. Memuat dataset Telco Customer Churn.
    2. Membersihkan kolom TotalCharges (ubah ke numerik & isi NaN dengan 0).
    3. Memisahkan fitur (X) dan target (y).
    4. Membagi data menjadi train dan test (80:20).
    
    Returns:
        X_train, X_test, y_train, y_test (DataFrame/Series)
    """
    
    # --- 1. Load Dataset ---
    # Mengambil data langsung dari repository IBM sesuai notebook eksperimen
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    # --- 2. Preprocessing / Cleaning ---
    # Mengubah TotalCharges menjadi angka (numeric), error akan jadi NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Mengisi nilai NaN pada TotalCharges dengan 0 (sesuai langkah di notebook Anda)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # --- 3. Splitting Data ---
    # Pisahkan Fitur (X) dan Target (y)
    # Target adalah kolom 'Churn', sisanya adalah Fitur
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Bagi data: 80% Training, 20% Testing
    # Random state 42 digunakan agar hasil split konsisten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Preprocessing Selesai!")
    print(f"Dimensi X_train: {X_train.shape}")
    print(f"Dimensi X_test : {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# --- Blok Main (Agar file ini bisa dites langsung) ---
if __name__ == "__main__":
    # Jalankan fungsi saat file ini dieksekusi langsung
    clean_and_split_data()