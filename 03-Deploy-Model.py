# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Deploying our Chat Model and enabling Online Evaluation Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-0.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's now deploy our model as an endpoint to be able to send real-time queries.
# MAGIC
# MAGIC Once our model is live, we will need to monitor its behavior to detect potential anomaly and drift over time. 
# MAGIC
# MAGIC We won't be able to measure correctness as we don't have a ground truth, but we can track model perplexity and other metrics like profesionalism over time.
# MAGIC
# MAGIC This can easily be done by turning on your Model Endpoint Inference table, automatically saving every query input and output as one of your Delta Lake tables.

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.12.0 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper_code/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Deploy our model with Inference tables
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-1.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC Let's start by deploying our model endpoint.
# MAGIC
# MAGIC Simply define the `auto_capture_config` parameter during the deployment (or through the UI) to define the table where the endpoint request payload will automatically be saved.
# MAGIC
# MAGIC Databricks will fill the table for you in the background, as a fully managed service.

# COMMAND ----------

import urllib
import json
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# COMMAND ----------

environment_vars={"DATABRICKS_TOKEN": "{{secrets/" + scope_name + "/rag_sp_token}}"}
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

w = WorkspaceClient()
serving_client = EndpointApiClient()

#### Start the endpoint using the REST API (you can do it using the UI directly)
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, model_name=model_name, model_version = version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True, auto_capture_config=auto_capture_config, environment_vars=environment_vars)

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

df_split = DataframeSplitInput(columns=["messages"],
                               data=[[ {"messages": [{"role": "user", "content": "What is Apache Spark?"}, 
                                                     {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
                                                     {"role": "user", "content": "Does it support streaming?"}
                                                    ]}]])
w = WorkspaceClient()
w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Let's give it a try, using Gradio as UI!
# MAGIC
# MAGIC All you now have to do is deploy your chatbot UI. Here is a simple example using Gradio ([License](https://github.com/gradio-app/gradio/blob/main/LICENSE)). Explore the chatbot gradio [implementation](https://huggingface.co/spaces/databricks-demos/chatbot/blob/main/app.py).
# MAGIC
# MAGIC *Note: this UI is hosted and maintained by Databricks for demo purpose and is not intended for production use. We'll soon show you how to do that with Lakehouse Apps!*

# COMMAND ----------

display_gradio_app("databricks-demos-chatbot")
