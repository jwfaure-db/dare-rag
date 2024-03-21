# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=4148934117703857&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1">

# COMMAND ----------

aws_account_id = "905790255208"
dbName = db = "default"
catalog = "catalog_" + aws_account_id
scope_name = "scope_" + aws_account_id
DOCS_S3_LOCATION = "s3://workshop-bucket-" + aws_account_id + "/documents"
VECTOR_SEARCH_ENDPOINT_NAME = "vs-endpoint-demo"
