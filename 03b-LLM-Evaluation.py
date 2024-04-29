# Databricks notebook source
# MAGIC %md
# MAGIC # (Optional) Lab 4: LLM evaluation with Databricks Monitoring
# MAGIC Let's now evaluate and monitor our model using the data recorded in inference tables . Here are the required steps:
# MAGIC
# MAGIC - Make sure the inference table is enabled (it was automatically setup in the previous cell)
# MAGIC - Consume all the inference table payload, and measure the model answer metrics (perplexity, complexity, etc)
# MAGIC - Save the result in your metric table. This can first be used to plot the metrics over time
# MAGIC - Leverage Databricks Lakehouse Monitoring  to analyze the metric evolution over time
# MAGIC
# MAGIC ## 4.1 Install dependencies and run helper code
# MAGIC
# MAGIC To get started, let's run the following code blocks within your new notebook. This installs dependencies, set environment variables, and run the helper code loaded into our Databricks Workspace. 

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.18.0 mlflow==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper_code/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Offline LLM evaluation
# MAGIC We will start with offline evaluation, scoring our model before its deployment. This requires a set of questions we want to ask to our model.
# MAGIC
# MAGIC In our case, we are fortunate enough to have a labeled training set (questions+answers) with state-of-the-art technical answers from our Databricks support team. Let's leverage it so we can compare our RAG predictions and ground-truth answers in MLflow.
# MAGIC
# MAGIC *Note: This is optional! We can benefit from the LLMs-as-a-Judge approach without ground-truth labels. This is typically the case if you want to evaluate "live" models answering any customer questions*

# COMMAND ----------

volume_folder =  volume_folder = f"/Volumes/{catalog}/{db}/pdf_volume/evaluation_dataset"
#Load the eval dataset from the repository to our volume
upload_dataset_to_volume(volume_folder)

# COMMAND ----------

spark.sql(f'''
CREATE OR REPLACE TABLE evaluation_dataset AS
  SELECT q.id, q.question, a.answer FROM parquet.`{volume_folder}/training_dataset_question.parquet` AS q
    LEFT JOIN parquet.`{volume_folder}/training_dataset_answer.parquet` AS a
      ON q.id = a.question_id ;''')

# COMMAND ----------

from databricks.sdk.service.serving import DataframeSplitInput

serving_endpoint_name = f"{catalog}_{db}_endpoint"

#### Let's generate some traffic to our endpoint. We send 50 questions and wait for them to be in our inference table
send_requests_to_endpoint_and_wait_for_payload_to_be_available(serving_endpoint_name, spark.table("evaluation_dataset").select('question'), limit=20)

# COMMAND ----------

# MAGIC %md
# MAGIC ##LLMs-as-a-judge: automated LLM evaluation with out of the box and custom GenAI metrics
# MAGIC
# MAGIC Since MLflow 2.8, there are many out of the box GenAI metrics and enables us to make our own GenAI metrics:
# MAGIC - Mlflow will automatically compute relevant task-related metrics. In our case, `model_type='question-answering'` will add the `toxicity` and `token_count` metrics.
# MAGIC - Then, we can import out of the box metrics provided by MLflow 2.8. Let's benefit from our ground-truth labels by computing the `answer_correctness` metric. 
# MAGIC - Finally, we can define customer metrics. Here, creativity is the only limit. In our demo, we will evaluate the `professionalism` of our Q&A chatbot.
# MAGIC
# MAGIC Online evaluation requires a couple steps to unpack the inference table output, compute the LLM metrics and turn on the Lakehouse Monitoring. Databricks provides a ready-to-use notebook (llm-inference-table-monitor.py ) that you can run directly to extract the data and setup the monitoring. Open the notebook for more details.
# MAGIC
# MAGIC Note that depending of your model input/output, you might need to change the notebook unpacking logic. See the notebook comments for more details.

# COMMAND ----------

monitor = dbutils.notebook.run("./helper_code/Inference-Tables-Analysis", 600, 
                            {"endpoint": serving_endpoint_name, 
                              "checkpoint_location": f'dbfs:/Volumes/{catalog}/{db}/pdf_volume/checkpoints/payload_metrics'}
                              )


# COMMAND ----------

url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/sql/dashboards/{json.loads(monitor)["dashboard_id"]}'
print(f"You can monitor the performance of your chatbot at {url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC In this Lab, you learned how to automate analysis and monitoring of generative AI applications with Databricks. You investigated the use of custom LLM metrics to track model performance over time.
# MAGIC
# MAGIC **In practice, consider adding a human feedback loop, reviewing where your model doesn't perform well. For example, by providing your customer simple way to flag incorrect answers. This is also a good opportunity to either improve your documentation or adjust your prompt, and ultimately add the correct answer to your evaluation dataset.**
# MAGIC
