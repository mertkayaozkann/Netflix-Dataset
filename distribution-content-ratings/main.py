import kaggle
import os
import pandas as pd
import matplotlib.pyplot as plt
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

    # 'rating' ve 'type' sütunlarını kontrol ediyoruz
    print(df[['rating', 'type']].head())

    # Null değerleri temizliyoruz
    df = df.dropna(subset=['rating', 'type'])

    return df

# Analyze content type with Spark
def analyze_content_type(data_path):
    # Spark oturumunu başlatıyoruz
    spark = SparkSession.builder \
        .appName("Netflix Content Type Analysis") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Veriyi Spark DataFrame olarak yüklüyoruz
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)

    # Spark ile içerik türlerinin dağılımını hesaplıyoruz
    content_type_counts = df_spark.groupBy("type").count().toPandas()

    # Sonuçları görselleştiriyoruz (Pie Chart)
    content_type_counts.plot(kind='pie', figsize=(8, 5), y='count', labels=content_type_counts['type'],
                             autopct='%1.1f%%', color=['blue', 'orange'])
    plt.title('Distribution of Content Types on Netflix')
    plt.xlabel('Content Type')
    plt.ylabel('Number of Titles')
    plt.xticks(rotation=0)
    plt.show()

# Ana fonksiyonları çalıştırma
def main():
    # Veriyi indiriyoruz
    file_path = download_data()

    # Preprocessing işlemi yapıyoruz
    df = preprocess_data(file_path)

    # Spark ile analiz yapıyoruz
    analyze_content_type(file_path)

if __name__ == "__main__":
    main()
