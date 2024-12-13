import kaggle
import os
import pandas as pd
from pyspark.sql import SparkSession
from kaggle.api.kaggle_api_extended import KaggleApi
from pyspark.sql.functions import trim

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
    df = df.dropna(subset=['listed_in'])

    # 'listed_in' sütununu kontrol ediyoruz
    print(df[['listed_in']].head())

    return df

# Popularity of Genres
def analyze_popularity_of_genres(data_path):
    # Spark oturumunu başlatıyoruz
    spark = SparkSession.builder \
        .appName("Netflix Popularity of Genres") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Veriyi Spark DataFrame olarak yüklüyoruz
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)

    # 'listed_in' sütunundaki genre'leri ayırıyoruz
    from pyspark.sql.functions import explode, split
    df_spark = df_spark.withColumn('genre', explode(split(df_spark['listed_in'], ',')))

    # Genre isimlerindeki boşlukları temizliyoruz
    df_spark = df_spark.withColumn('genre', trim(df_spark['genre']))

    # Genre'leri sayıyoruz
    genre_counts = df_spark.groupBy('genre').count().toPandas()

    # Veri manipülasyonu ve temizleme
    genre_counts.columns = ['Genre', 'Count']

    # Boş değerleri temizle
    genre_counts = genre_counts.dropna(subset=['Genre'])

    # En popüler genre'leri sıralıyoruz
    genre_counts_sorted = genre_counts.sort_values(by='Count', ascending=False)

    # Pandas ile tabloyu yazdırıyoruz (ilk 10 genre)
    print(genre_counts_sorted.head(10))

# Ana fonksiyonları çalıştırma
def main():
    # Veriyi indiriyoruz
    file_path = download_data()

    # Preprocessing işlemi yapıyoruz
    df = preprocess_data(file_path)

    # Spark ile analiz yapıyoruz
    analyze_popularity_of_genres(file_path)

if __name__ == "__main__":
    main()
