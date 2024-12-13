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
    df = df.dropna(subset=['cast'])

    # 'cast' sütununu kontrol ediyoruz
    print(df[['cast']].head())

    return df

# Popularity of Actors
def analyze_popularity_of_actors(data_path):
    # Spark oturumunu başlatıyoruz
    spark = SparkSession.builder \
        .appName("Netflix Popularity of Actors") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Veriyi Spark DataFrame olarak yüklüyoruz
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)

    # 'cast' sütunundaki oyuncuları ayırıyoruz
    from pyspark.sql.functions import explode, split
    df_spark = df_spark.withColumn('actor', explode(split(df_spark['cast'], ',')))

    # Oyuncu isimlerindeki boşlukları temizliyoruz
    df_spark = df_spark.withColumn('actor', trim(df_spark['actor']))

    # Oyuncuları sayıyoruz
    actor_counts = df_spark.groupBy('actor').count().toPandas()

    # Veri manipülasyonu ve temizleme
    actor_counts.columns = ['Actor', 'Count']

    # Boş değerleri temizle
    actor_counts = actor_counts.dropna(subset=['Actor'])

    # En popüler oyuncuları sıralıyoruz
    actor_counts_sorted = actor_counts.sort_values(by='Count', ascending=False)

    # Pandas ile tabloyu yazdırıyoruz (ilk 10 oyuncu)
    print(actor_counts_sorted.head(10))

# Ana fonksiyonları çalıştırma
def main():
    # Veriyi indiriyoruz
    file_path = download_data()

    # Preprocessing işlemi yapıyoruz
    df = preprocess_data(file_path)

    # Spark ile analiz yapıyoruz
    analyze_popularity_of_actors(file_path)

if __name__ == "__main__":
    main()
