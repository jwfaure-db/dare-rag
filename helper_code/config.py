# Databricks notebook source

# COMMAND ----------

# DATABRICKS PAT
access_token = "{Fill in your Databricks Personal Access Token (PAT)}"

# AWS CONFIGURATION
aws_account_id = "{Fill in your AWS Account ID}"
aws_access_key = "{Fill in your AWS Access Key}"
aws_secret_access_key = "{Fill in your AWS Secret Access Key}"

# DATABRICKS VECTOR SEARCH ENDPOINT
VECTOR_SEARCH_ENDPOINT_NAME = "vector-search-endpoint" 
# (You may have to change the above value if instructed)

# DATABRICKS EXTERNAL LOCATION S3 URL
S3_LOCATION = "{Ignore at AWS-hosted events}"

# DATABRICKS CONFIGURATION
catalog = "catalog_" + aws_account_id
dbName = db = "default"
scope_name = "scope_" + aws_account_id
workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# DATABRICKS MODEL SERVING CONFIGURATION
embeddings_model_endpoint_name = "embeddings_" + aws_account_id
bedrock_chat_model_endpoint_name = "claude_sonnet_" + aws_account_id

# COMMAND ----------

# Automate identification of external location for AWS-hosted events
workshop_prefix = "s3://db-workshop-" + aws_account_id

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
external_locations = spark.sql("SHOW EXTERNAL LOCATIONS").collect()
if len(external_locations) > 0:
    for row in external_locations:
        if row.url.startswith(workshop_prefix):
            S3_LOCATION = row.url
            print("Configured S3_LOCATION: " + S3_LOCATION)
            

# COMMAND ----------

# MAGIC %run ./00-create-secrets

# COMMAND ----------

scopes = dbutils.secrets.listScopes()

if scope_name not in [scope.name for scope in scopes]:
    create_scope(scope_name, access_token, workspace_url)
    create_secret("rag_sp_token", access_token, scope_name, access_token, workspace_url)
    create_secret("aws_access_key_id", aws_access_key, scope_name, access_token, workspace_url)
    create_secret("aws_secret_access_key", aws_secret_access_key, scope_name, access_token, workspace_url)