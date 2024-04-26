# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=4148934117703857&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1">

# COMMAND ----------

# AWS CONFIGURATION FROM CLOUDFORMATION
S3_LOCATION = "s3://{Fill in your S3 Bucket}"
aws_account_id = "{Fill in your aws account ID from your Event Outputs / AWS CloudFormation output}"
aws_access_key = "{Fill in aws access key from Event Outputs / AWS CloudFormation output}"
aws_secret_access_key = "{Fill in aws secret access key from your Event Outputs / AWS CloudFormation output}"

# DATABRICKS CONFIGURATION
access_token = "{Fill in Databricks Personal Access Token you generated}"
catalog = "{Fill in name of the catalog that you created in Account Setup}"
# catalog = "catalog_" + aws_account_id
dbName = db = "default"
scope_name = "scope_" + aws_account_id
workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")


# MODEL ENDPOINT CONFIGURATION
embeddings_model_endpoint_name = "embeddings_" + aws_account_id
bedrock_chat_model_endpoint_name = "claude_sonnet_" + aws_account_id
VECTOR_SEARCH_ENDPOINT_NAME = "{instructor to provide vector search endpoint name for workshop}"

# COMMAND ----------

# MAGIC %run ./00-create-secrets

# COMMAND ----------

scopes = dbutils.secrets.listScopes()
if scope_name not in [scope.name for scope in scopes]:
    create_scope(scope_name, access_token, workspace_url)
    create_secret("rag_sp_token", access_token,
                  scope_name, access_token, workspace_url)
    create_secret("aws_access_key_id", aws_access_key,
                  scope_name, access_token, workspace_url)
    create_secret("aws_secret_access_key", aws_secret_access_key,
                  scope_name, access_token, workspace_url)
