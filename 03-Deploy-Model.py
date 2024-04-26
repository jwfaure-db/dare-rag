# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Lab 3 Deploy your model with inference tables
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Getting started
# MAGIC To get started, install Python libraries, and restart the kernel to use updated packages.

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.18.0 mlflow==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's run the init notebook, which imports helpers for the lab.

# COMMAND ----------

# MAGIC %run ./helper_code/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3.2 Deploy our model with Inference tables
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-1.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC Let's start deploying your model serving endpoint.
# MAGIC
# MAGIC 1. Let's first import some pre-requisite libraries.

# COMMAND ----------

from databricks.sdk.service.serving import DataframeSplitInput
import urllib
import json
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Let's configure the variables for the model to deploy.
# MAGIC
# MAGIC Enable inference tables on model serving endpoints for monitoring and debugging
# MAGIC The auto_capture_config parameter defines the table where the endpoint request payload will automatically be saved. Databricks will fill the table for you in the background, as a fully managed service.

# COMMAND ----------

environment_vars = {"DATABRICKS_TOKEN": dbutils.secrets.get(
    scope_name, "rag_sp_token")}
model_name = f"{catalog}.{db}.advanced_chatbot_model"
serving_endpoint_name = f"{catalog}_{db}_endpoint"[:63]

client = MlflowClient()
mlflow.set_registry_uri('databricks-uc')
latest_model = client.get_registered_model(model_name)
version = get_latest_model_version(model_name)

auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": db,
    "table_name_prefix": serving_endpoint_name
}

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Now, you're ready to deploy your model.
# MAGIC
# MAGIC **Please be patient.**
# MAGIC **It may take up to 15 to 30 minutes for Databricks to deploy your model.**

# COMMAND ----------

w = WorkspaceClient()
serving_client = EndpointApiClient()

# Start the endpoint using the REST API (you can do it using the UI directly)
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, model_name=model_name, model_version=version, workload_size="Small",scale_to_zero_enabled=True, wait_start=True, auto_capture_config=auto_capture_config, environment_vars=environment_vars)

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC Once deployed, let's try sending a query to your chatbot.

# COMMAND ----------


df_split = DataframeSplitInput(columns=["messages"],
                               data=[[{"messages": [{"role": "user", "content": "What is Apache Spark?"},
                                                    {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."},
                                                    {"role": "user",
                                                     "content": "Does it support streaming?"}
                                                    ]}]])
w = WorkspaceClient()
w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC Great job! As you can see, Databricks makes it easy to deploy your own chat model as an API with built-in monitoring. Let's continue.
# MAGIC
# MAGIC Now, let's now build a user interface for our chatbot. As part of your workshop AWS account, you have a [Streamlit](https://streamlit.io/) chatbot application deployed to an Amazon EC2 instance. If you're not familiar, Streamlit is a way to quickly turn data scripts into web applications using Python.
# MAGIC
# MAGIC Let's head back to Workshop Studio Documentation to see how we can integrate our model endpoint with a web application on AWS
