from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DecimalType, FloatType, ByteType, ShortType, LongType, NumericType
from pyspark.sql.functions import col, count, countDistinct, when, first, lit, expr, percentile, radians, asin, sin, cos, sqrt, split, atan2
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Row
from decimal import Decimal, ROUND_HALF_UP
from pyspark.sql import SparkSession
from scipy.stats import entropy
import numpy as np
import math


### STEP 1: Mounting Data Lake Gen2 Using Service Principle to Databricks


class AzureStorageManager:
    def __init__(self, client_id, client_secret, tenant_id):
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.configs = {
            "fs.azure.account.auth.type": "OAuth",
            "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
            "fs.azure.account.oauth2.client.id": self.client_id,
            "fs.azure.account.oauth2.client.secret": self.client_secret,
            "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token"
        }

    def mount_storage_account(self, folder_name, storage_account_name, mount_point):
        source = f"abfss://{folder_name}@{storage_account_name}.dfs.core.windows.net/"

        # Check if the source is already mounted
        mounted_sources = [mount.source for mount in dbutils.fs.mounts()]
        if not source in mounted_sources:
            dbutils.fs.mount(source=source, mount_point=mount_point, extra_configs=self.configs)
            print("Source mounted")
        else:
            print("Source already mounted")

    def list_mounted_folders(self, mount_point):
        list_mounted_folders = dbutils.fs.ls(mount_point)
        return [folder.name for folder in list_mounted_folders]



class CustomAzureStorageManager(AzureStorageManager):
    
    def __init__(self, client_id, client_secret, tenant_id, folder_name, storage_account_name, mount_point):
        # Call the constructor of the parent class using super()
        super().__init__(client_id, client_secret, tenant_id)
        self.folder_name = folder_name
        self.storage_account_name = storage_account_name
        self.mount_point = mount_point

    def mount_and_list(self):
        # Mount the storage account
        self.mount_storage_account(self.folder_name, self.storage_account_name, self.mount_point)

        # List all folders in the storage account
        mounted_folders = self.list_mounted_folders(self.mount_point)
        for folder in mounted_folders:
            print(folder)

    def mount_specific_folder(self, nested_folder_name):
        # Mount the storage account
        exact_mount_point = str(self.mount_point) +  f"{nested_folder_name}"

        list_mounted_files = dbutils.fs.ls(exact_mount_point)
        spark = SparkSession.builder.getOrCreate()
        rows = [Row(file_name=file.name, file_path=file.path) for file in list_mounted_files]

        df = spark.createDataFrame(rows)

        return df


client_id = dbutils.secrets.get(scope='azurestorageaccount97-Scope', key='SparkExplorationClientId')
tenant_id = dbutils.secrets.get(scope='azurestorageaccount97-Scope', key='SparkExplorationTenantId')
client_secret = dbutils.secrets.get(scope='azurestorageaccount97-Scope', key='SparkExplorationClientSecretId')
folder_name = dbutils.secrets.get(scope='azurestorageaccount97-Scope', key='SEFolderName')
storage_account_name = dbutils.secrets.get(scope='azurestorageaccount97-Scope', key='StorageAccountName')
mount_point = dbutils.secrets.get(scope='azurestorageaccount97-Scope', key='SEMountPoint')

# Create an instance of CustomAzureStorageManager
custom_storage_manager = CustomAzureStorageManager(
    client_id, client_secret, tenant_id, folder_name, storage_account_name, mount_point
)


# Mount and list the storage account
custom_storage_manager.mount_and_list()


# Mount and list the specific files
generate_file_metadata = custom_storage_manager.mount_specific_folder("raw")
display(generate_file_metadata)


### STEP 2: load Data from  Data Lake Gen2 

class CSVReader:
    def __init__(self, storage_account_name, folder_name):
        self.storage_account_name = storage_account_name
        self.folder_name = folder_name
        self.spark = SparkSession.builder.appName("CSVReader").getOrCreate()

    def defined_schema(self):
        circuits_schema = StructType([
            StructField("CustomerID", IntegerType(), True),
            StructField("Customer_Lifetime_Value", FloatType(), True),
            StructField("Coverage", StringType(), True),
            StructField("Education", StringType(), True),
            StructField("EmploymentStatus", StringType(), True),
            StructField("Gender", StringType(), True),
            StructField("Income", IntegerType(), True),
            StructField("Location_Geo", StringType(), True),
            StructField("Location_Code", StringType(), True),
            StructField("Marital_Status", StringType(), True),
            StructField("Monthly_Premium_Auto", IntegerType(), True),
            StructField("Months_Since_Last_Claim", IntegerType(), True),
            StructField("Months_Since_Policy_Inception", IntegerType(), True),
            StructField("Number_of_Open_Complaints", IntegerType(), True),
            StructField("Number_of_Policies", IntegerType(), True),
            StructField("Policy_Type", StringType(), True),
            StructField("Policy", StringType(), True),
            StructField("Renew_Offer_Type", StringType(), True),
            StructField("Sales_Channel", StringType(), True),
            StructField("Total_Claim_Amount",FloatType(), True),
            StructField("Vehicle_Class", StringType(), True),
            StructField("Vehicle_Size", IntegerType(), True)
        ])
        return circuits_schema

    def read_csv(self, file_name, header=True):
        full_path = f"dbfs:/mnt/{self.storage_account_name}/{self.folder_name}/{file_name}"
        print(f"Reading CSV from: {full_path}")
        clv_schema = self.defined_schema()
        clv_data = self.spark.read.csv(full_path, schema = clv_schema, header=header)
        return clv_data
    
    def write_csv(self, data, write_file_loation, header=True):
        full_path = f"dbfs:/mnt/{self.storage_account_name}/{self.folder_name}/{write_file_loation}"
        print(f"Writing CSV to: {full_path}")
        clv_schema = self.defined_schema()
        #clv_data = self.spark.read.csv(full_path, schema = clv_schema, header=header)
        return data.write.csv(full_path, header=header)
    

file_name = "/raw/Customer_Live_Time_Value.csv"
write_file_loation = "/processed/Customer_Live_Time_Value.csv"

csv_reader = CSVReader(storage_account_name, folder_name)
clv_data = csv_reader.read_csv(file_name, header=True)
display(clv_data)



### STEP 3: Data Exploration


class Attribute_Information:

    def __init__(self):
        print("Attribute Information object created")
        self.spark = SparkSession.builder.appName("Attribute_Information").getOrCreate()
    
    def entropy_udf(self,column):

        """
        User-defined function (UDF) to calculate the entropy for a column.
        """
        value_counts = column.value_counts()
        probabilities = value_counts / value_counts.sum()
        return entropy(probabilities, base=2)

    def round_decimal(self,value):

        """
        Round float value to Decimal with 6 decimal places.
        """
        return Decimal(str(value)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def Column_information(self, df):
        """
        This method will give us a basic
        information of the dataframe like
        Count of Attributes, Count of rows,
        Numerical Attributes, Categorical
        Attributes, Factor Attributes, etc.
        """
        data_info = self.spark.createDataFrame(
            [(df.count(), len(df.columns))],
            ['No of observation', 'No of Variables']
        )

        dtypes_df = self.spark.createDataFrame(df.dtypes, ['column', 'type'])
        numeric_vars = dtypes_df.filter(dtypes_df['type'].isin(['int', 'bigint', 'float', 'double'])).count()
        data_info = data_info.withColumn('No of Numerical Variables', lit(numeric_vars))

        categorical_vars = dtypes_df.filter(dtypes_df['type'] == 'string').count()
        data_info = data_info.withColumn('No of Categorical Variables', lit(categorical_vars))

        boolean_vars = dtypes_df.filter(dtypes_df['type'] == 'boolean').count()
        data_info = data_info.withColumn('No of Logical Variables', lit(boolean_vars))

        date_vars = dtypes_df.filter(dtypes_df['type'] == 'date').count()
        data_info = data_info.withColumn('No of Date Variables', lit(date_vars))

        zero_variance_cols = df.select([countDistinct(col(col_name)).alias(col_name) for col_name in df.columns]).collect()
        zero_variance_cols = sum(1 for row in zero_variance_cols[0] if row == 1)
        data_info = data_info.withColumn('No of zero variance variables', lit(zero_variance_cols))


        return data_info
    

    def Agg_Tabulation(self, data):

        """
        This method is an extension of schema that gives additional
        information about the data like Entropy value, Missing 
        Value Percentage, and some observations.
        """

        total_rows = data.count()

        schema = StructType([
            StructField("Column_Name", StringType(), True),
            StructField("Data_Type", StringType(), True),
            StructField("missing_count", IntegerType(), True),
            StructField("unique_value_count", IntegerType(), True),
            StructField("missing_percentage", IntegerType(), True),
            StructField("entropy", DecimalType(10, 6), True)
        ])
        print("=" * 110)
        print("Aggregation of Table")
        print("=" * 110)

        Dtypes = spark.createDataFrame(data.dtypes, ["Column_Name", "Data_Type"])
        missing_count = [data.filter(data[col].isNull()).count() for col in data.columns]
        unique_value_count = [data.select(col).distinct().count() for col in data.columns]
        missing_percentage = [round(count / total_rows * 100) for count in missing_count]
        entropy_values = [self.entropy_udf(data.select(col).toPandas()[col]) for col in data.columns]
        entropy_values = [self.round_decimal(value) for value in entropy_values]

        columns_with_counts = zip(Dtypes.collect(), missing_count, unique_value_count, missing_percentage, entropy_values)
        rows = [Row(Column_Name=col[0], Data_Type=col[1], missing_count=mc, unique_value_count=uc, missing_percentage=mp, entropy=ent) for (col, mc, uc, mp, ent) in columns_with_counts]

        data_with_schema = spark.createDataFrame(rows, schema)

        return data_with_schema
    

    def get_numeric_columns(self, dataframe):

        """
        Get a list of column names containing numeric data types from a DataFrame.
        
        Args:
            dataframe (DataFrame): The input DataFrame.
            
        Returns:
            list: List of column names with numeric data types.
        """
        numeric_columns = []
        for column in dataframe.columns:
            data_type = dataframe.schema[column].dataType
            if isinstance(data_type, (IntegerType, DoubleType, ByteType, ShortType, LongType, FloatType)):
                numeric_columns.append(column)
        
        return dataframe.select(*numeric_columns)

    
    def statistical_summary(self, df):

        """
        This method will return various percentiles
        of the data including count and mean
        """
        
        #numeric_cols = [col_name for col_name, col_type in df.dtypes if col_type in ('int', 'bigint', 'float', 'double')]
        numeric_cols =  self.get_numeric_columns(df)
        df_num = df.select(*numeric_cols)

        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        
        data_stat_num = df_num.summary(*['count', 'mean', 'stddev', 'min', *["{}%".format(int(p * 100)) for p in percentiles], 'max'])

        return data_stat_num

    def calculate_total_outlier_count(self,df):

        """
        Calculate the total count of outliers for numeric columns in a DataFrame using PySpark.

        :param df: Input DataFrame with specified schema
        :return: List of total outlier counts for each numeric column
        """
        #numeric_columns =  self.get_numeric_columns(df)
        numeric_columns = [col_name for col_name, col_type in df.dtypes if isinstance(df.schema[col_name].dataType, (IntegerType, DoubleType, FloatType))]
        total_outlier_counts = ['Outlier_Count']

        for column_name in numeric_columns:
            
            q1_values = df.approxQuantile(column_name, [0.25], 0.01)
            q3_values = df.approxQuantile(column_name, [0.75], 0.01)
            
            if q1_values and q3_values:
                
                q1 = q1_values[0]
                q3 = q3_values[0]
                iqr = q3 - q1


                upper_out = q3 + 1.5 * iqr
                lower_out = q1 - 1.5 * iqr

                outlier_count = df.filter((col(column_name) > upper_out) | (col(column_name) < lower_out)).count()
                total_outlier_counts.append(outlier_count)
            else:
                total_outlier_counts.append(0) 
            
            f_data = spark.createDataFrame([total_outlier_counts])

        return f_data



Info = Attribute_Information()

Column_information = Info.Column_information(clv_data)
display(Column_information)


Agg_Tabulation = Info.Agg_Tabulation(clv_data)
display(Agg_Tabulation)


display(Info.statistical_summary(clv_data).union(Info.calculate_total_outlier_count(clv_data)))



class Compute_Haversine_Distance():
    def __init__(self):
        print("Distance object created")
        
    
    def haversine_distance(self,df):

        """
        This part is created to split
        the Location Geo variable into
        Lati and Longi by comma separated
        values to compute distance
        """

        df = df.withColumn("Location_Geo", split(df["Location_Geo"], ","))
        df = df.withColumn("Lati", df["Location_Geo"].getItem(0).cast("double"))
        df = df.withColumn("Longi", df["Location_Geo"].getItem(1).cast("double"))

        """
        This part helps to compute the 
        distance between the points of 
        latitude and longitude by replicating
        the Haversine formula
        """
        

        lat_ref = math.radians(37.2175900)
        lon_ref = math.radians(-56.7213600)
        
        df = df.withColumn("LAT_rad", radians(df["Lati"]))
        df = df.withColumn("LON_rad", radians(df["Longi"]))

        df = df.withColumn("dlon", df["LON_rad"] - lon_ref)
        df = df.withColumn("dlat", df["LAT_rad"] - lat_ref)

        df = df.withColumn("haversine_distance",
                        asin(lit(sqrt(lit(sin(col("dlat")/2)**2 +\
                            cos(radians(lit(lat_ref)))* cos(col("LAT_rad")) *\
                                sin(col("dlon")/2 **2) ))))
                    )

        return df




haversine = Compute_Haversine_Distance()

clv_data = haversine.haversine_distance(clv_data)
display(clv_data)



class Categorical_Imputer():
    
    def __init__(self):
        print("Imputer object created")
        

    def Impute_categorical_columns(self, df):

        df = df.replace("NA", 'null')
        string_columns = [col_name for col_name, col_type in df.dtypes if isinstance(df.schema[col_name].dataType, (StringType))]
        column_to_remove = "Location_Geo"

        if column_to_remove in string_columns:
            string_columns.remove(column_to_remove)

        for col_name in string_columns:

            top_mode = df.groupBy(col_name).agg(count('*').alias('count')).orderBy(col('count').desc()).select(col_name).limit(1).first()[col_name]

            filled_df = df.replace("null", top_mode)

        return filled_df



Imputer = Categorical_Imputer()
clv_data = Imputer.Impute_categorical_columns(clv_data)
display(clv_data)



clv_data = clv_data.drop("Location_Geo")
display(clv_data)


### STEP 3: Writing Data to Data Lake Gen 2


csv_reader.write_csv(clv_data, write_file_loation, header=True)