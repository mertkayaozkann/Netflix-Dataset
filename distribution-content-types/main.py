import kaggle
import os
import pandas as pd
from pyspark.sql import SparkSession
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle API'sini başlatıyoruz
def download_data():
    # Kaggle API'sini başlat
    api = KaggleApi()
    api.authenticate()

    # Dataset'in adı
    dataset_name = 'shivamb/netflix-shows'
    output_dir = '/app/dataset'
    os.makedirs(output_dir, exist_ok=True)

    # Veri setini Kaggle API üzerinden indiriyoruz
    api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
    print("Veri başarıyla indirildi!")
    return os.path.join(output_dir, 'netflix_titles.csv')


# Preprocessing işlemi yapıyoruz
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Null değerleri temizliyoruz
    df = df.dropna(subset=['country', 'type'])

    # 'country' ve 'type' sütunlarını kontrol ediyoruz
    print(df[['country', 'type']].head())

    return df

# Analyze content type by country
def analyze_content_type_by_country(data_path):
    # Spark oturumunu başlatıyoruz
    spark = SparkSession.builder \
        .appName("Netflix Content Type Analysis by Country") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Veriyi Spark DataFrame olarak yüklüyoruz
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)

    # 'country' ve 'type' sütunlarına göre grup oluşturup sayma işlemi
    type_counts_by_country = df_spark.groupBy(['country', 'type']).count().toPandas()

    # Veri manipülasyonu
    type_counts_by_country.columns = ['Country', 'Content Type', 'Count']
    type_counts_by_country = type_counts_by_country.pivot(index='Country', columns='Content Type', values='Count')

    # NaN değerleri sıfırla değiştir
    type_counts_by_country = type_counts_by_country.fillna(0)

    # Toplam sayıyı hesapla
    type_counts_by_country['Total'] = type_counts_by_country.sum(axis=1)

    # Sonuçları sıralıyoruz
    type_counts_by_country_sorted = type_counts_by_country.sort_values(by='Total', ascending=False)

    # Pandas ile tabloyu yazdırıyoruz
    print(type_counts_by_country_sorted[['Movie', 'TV Show', 'Total']].head())  # İlk 5 satırı yazdır

# Ana fonksiyonları çalıştırma
def main():
    # Veriyi indiriyoruz
    file_path = download_data()

    # Preprocessing işlemi yapıyoruz
    df = preprocess_data(file_path)

    # Spark ile analiz yapıyoruz
    analyze_content_type_by_country(file_path)

if __name__ == "__main__":
    main()
