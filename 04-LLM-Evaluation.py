# Databricks notebook source
# MAGIC %md
# MAGIC # (Optional) Lab 4: LLM evaluation
# MAGIC
# MAGIC ## 4.1 Install dependencies and run helper code
# MAGIC
# MAGIC To get started, let's run the following code blocks within your new notebook. This installs dependencies, set environment variables, and run the helper code loaded into our Databricks Workspace. 

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 databricks-sdk==0.18.0 langchain==0.1.5 databricks-vectorsearch==0.22 textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 transformers==4.30.2 torch==2.0.1 cloudpickle==2.2.1 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper_code/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Offline LLM evaluation
# MAGIC We will start with offline evaluation, scoring our model before its deployment. This requires a set of questions we want to ask to our model.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-offline-0.png?raw=true" style="display: block; margin: 25px auto">
# MAGIC
# MAGIC In our case, we are fortunate enough to have a labeled training set (questions+answers) with state-of-the-art technical answers from our Databricks support team. Let's leverage it so we can compare our RAG predictions and ground-truth answers in MLflow.
# MAGIC
# MAGIC *Note: This is optional! We can benefit from the LLMs-as-a-Judge approach without ground-truth labels. This is typically the case if you want to evaluate "live" models answering any customer questions*

# COMMAND ----------

volume_folder =  volume_folder = f"/Volumes/{catalog}/{db}/pdf_volume/evaluation_dataset"
#Load the eval dataset from the repository to our volume
upload_dataset_to_volume(volume_folder)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparing our evaluation dataset

# COMMAND ----------

spark.sql(f'''
CREATE OR REPLACE TABLE evaluation_dataset AS
  SELECT q.id, q.question, a.answer FROM parquet.`{volume_folder}/training_dataset_question.parquet` AS q
    LEFT JOIN parquet.`{volume_folder}/training_dataset_answer.parquet` AS a
      ON q.id = a.question_id ;''')

display(spark.table('evaluation_dataset'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Automated Evaluation of our chatbot model registered in Unity Catalog
# MAGIC
# MAGIC Let's retrieve the chatbot model we registered in Unity Catalog and predict answers for each questions in the evaluation set.

# COMMAND ----------

import mlflow
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope_name, "rag_sp_token")
model_name = f"{catalog}.{db}.advanced_chatbot_model"
model_version_to_evaluate = get_latest_model_version(model_name)
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

@pandas_udf("string")
def predict_answer(questions):
    def answer_question(question):
        dialog = {"messages": [{"role": "user", "content": question}]}
        return rag_model.invoke(dialog)['result']
    return questions.apply(answer_question)

df_qa = (spark.read.table('evaluation_dataset')
                  .selectExpr('question as inputs', 'answer as targets')
                  .where("targets is not null")
                  .sample(fraction=0.005, seed=40)) #small sample for interactive demo

df_qa_with_preds = df_qa.withColumn('preds', predict_answer(col('inputs'))).cache()

display(df_qa_with_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ##LLMs-as-a-judge: automated LLM evaluation with out of the box and custom GenAI metrics
# MAGIC
# MAGIC Since MLflow 2.8, there are many out of the box GenAI metrics and enables us to make our own GenAI metrics:
# MAGIC - Mlflow will automatically compute relevant task-related metrics. In our case, `model_type='question-answering'` will add the `toxicity` and `token_count` metrics.
# MAGIC - Then, we can import out of the box metrics provided by MLflow 2.8. Let's benefit from our ground-truth labels by computing the `answer_correctness` metric. 
# MAGIC - Finally, we can define customer metrics. Here, creativity is the only limit. In our demo, we will evaluate the `professionalism` of our Q&A chatbot.
# MAGIC
# MAGIC Note that depending of your model input/output, you might need to change the notebook unpacking logic. See the notebook comments for more details.
# MAGIC
# MAGIC Online evaluation requires a couple steps to unpack the inference table output, compute the LLM metrics and turn on the Lakehouse Monitoring. Databricks provides a ready-to-use notebook [llm-inference-table-monitor.py](https://docs.databricks.com/en/_extras/notebooks/source/monitoring/llm-inference-table-monitor.html) that you can run directly to extract the data and setup the monitoring. Open the notebook for more details.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Claude on Amazon Bedrock as LLM Judge

# COMMAND ----------

client = mlflow.deployments.get_deploy_client("databricks")

if bedrock_chat_model_endpoint_name not in [endpoints['name'] for endpoints in client.list_endpoints()]:
    client.create_endpoint(
        name=bedrock_chat_model_endpoint_name,
        config={
            "served_entities": [
                {
                    "external_model": {
                        "name": "claude-3-sonnet-20240229-v1:0",
                        "provider": "amazon-bedrock",
                        "task": "llm/v1/chat",
                        "amazon_bedrock_config": {
                            "aws_region": "us-west-2",
                            "aws_access_key_id": "{{secrets/" + scope_name + "/aws_access_key_id}}",
                            "aws_secret_access_key": "{{secrets/" + scope_name + "/aws_secret_access_key}}",
                            "bedrock_provider": "anthropic",
                        },
                    }
                }
            ]
        },
    )
else:
    print("model already created")


endpoint_name = bedrock_chat_model_endpoint_name

# COMMAND ----------

from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Because we have our labels (answers) within the evaluation dataset, we can evaluate the answer correctness as part of our metric. Again, this is optional.
answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")
print(answer_correctness_metrics)

# COMMAND ----------

# Adding a custom professionalism metric

professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

print(professionalism)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start the evaluation run

# COMMAND ----------

from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

#This will automatically log all
with mlflow.start_run(run_name="chatbot_rag") as run:
    eval_results = mlflow.evaluate(data = df_qa_with_preds.toPandas(), # evaluation data,
                                   model_type="question-answering", # toxicity and token_count will be evaluated   
                                   predictions="preds", # prediction column_name from eval_df
                                   targets = "targets",
                                   extra_metrics=[answer_correctness_metrics, professionalism])
    
eval_results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization of our GenAI metrics produced by our LLM judge
# MAGIC
# MAGIC You can open your MLFlow experiment runs from the **Experiments** menu in the left navigation pane. From here, you can compare multiple model versions, and filter by correctness to spot where your model doesn't answer well.
# MAGIC
# MAGIC Based on that and depending on the issue, you can either fine tune your prompt, your model fine tuning instruction with RLHF, or improve your documentation.
# MAGIC
# MAGIC ### Custom visualizations
# MAGIC You can equaly plot the evaluation metrics directly from the run, or pulling the data from MLFlow:

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

import plotly.express as px
px.histogram(df_genai_metrics, x="token_count", labels={"token_count": "Token Count"}, title="Distribution of Token Counts in Model Responses")

# COMMAND ----------

# Counting the occurrences of each answer correctness score
px.bar(df_genai_metrics['answer_correctness/v1/score'].value_counts(), title='Answer Correctness Score Distribution')

# COMMAND ----------

df_genai_metrics['toxicity'] = df_genai_metrics['toxicity/v1/score'] * 100
fig = px.scatter(df_genai_metrics, x='toxicity', y='answer_correctness/v1/score', title='Toxicity vs Correctness', size=[10]*len(df_genai_metrics))
fig.update_xaxes(tickformat=".2f")

# COMMAND ----------

# MAGIC %md
# MAGIC ## This is looking good, let's tag our model as production ready
# MAGIC After reviewing the model correctness and potentially comparing its behavior to your other previous version, we can flag our model as ready to be deployed.
# MAGIC
# MAGIC Note: Evaluation can be automated and part of a MLOps step: once you deploy a new Chatbot version with a new prompt, run the evaluation job and benchmark your model behavior vs the previous version.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Live monitoring using Inference Tables
# MAGIC
# MAGIC Previously, you tried offline evaluation which can be done prior to deployment. Now, let's try online evaluation where you will be using [inference tables](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html). Here are the required steps:
# MAGIC
# MAGIC - Make sure the inference table is enabled (it was automatically setup in the previous cell)
# MAGIC - Consume all the inference table payload, and measure the model answer metrics (perplexity, complexity, etc)
# MAGIC - Save the result in your metric table. This can first be used to plot the metrics over time
# MAGIC - Leverage Databricks Lakehouse Monitoring  to analyze the metric evolution over time
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-1.png?raw=true" style="display: block; margin: 25px auto">
# MAGIC
# MAGIC Monitoring the performance of models in production workflows is an important aspect of the AI and ML model lifecycle. Inference Tables simplify monitoring and diagnostics for models by continuously logging serving request inputs and responses (predictions) from Databricks Model Serving endpoints and saving them into a Delta table in Unity Catalog. You can then use all of the capabilities of the Databricks platform, such as DBSQL queries, notebooks, and Lakehouse Monitoring to monitor, debug, and optimize your models.
# MAGIC
# MAGIC You can enable inference tables on any existing or newly created model serving endpoint, and requests to that endpoint are then automatically logged to a table in UC. 
# MAGIC
# MAGIC Online evaluation requires a couple steps to unpack the inference table output, compute the LLM metrics and turn on the Lakehouse Monitoring. Databricks provides a ready-to-use notebook ([llm-inference-table-monitor.py](https://docs.databricks.com/en/_extras/notebooks/source/monitoring/llm-inference-table-monitor.html)) that you can run directly to extract the data and setup the monitoring. Open the notebook for more details.

# COMMAND ----------

serving_endpoint_name = f"{catalog}_{db}_endpoint"

monitor = dbutils.notebook.run("./helper_code/Inference-Tables-Analysis", 600, {"endpoint": serving_endpoint_name, "checkpoint_location": f'dbfs:/Volumes/{catalog}/{db}/pdf_volume/checkpoints/payload_metrics'})

# COMMAND ----------

url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/sql/dashboards/{json.loads(monitor)["dashboard_id"]}'
print(f"You can monitor the performance of your chatbot at {url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC In this Lab, you learned how to automate analysis and monitoring of generative AI applications with Databricks. Evaluating your chatbot is key to measure your future version impact, and Databricks on AWS makes it easy leveraging automated Workflow for your MLOps pipelines. You investigated: 
# MAGIC
# MAGIC * Offline evaluation: How you can use custom metrics and LLM as a judge techniques to evaluate your model. For a production-grade GenAI application, this step should be automated and part as a job, executed everytime the model is changed and benchmarked against previous run to make sure you don't have performance regression.
# MAGIC
# MAGIC * Live monitoring: How you can use inference tables to monitor model performance over time. 
# MAGIC
# MAGIC * In practice, consider adding a human feedback loop, reviewing where your model doesn't perform well. For example, by providing your customer simple way to flag incorrect answers. This is also a good opportunity to either improve your documentation or adjust your prompt, and ultimately add the correct answer to your evaluation dataset.
# MAGIC
