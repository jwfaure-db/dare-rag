# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-josh_faure` from the dropdown menu ([open cluster configuration](https://dbc-7fdc927a-14d5.cloud.databricks.com/#setting/clusters/1008-093508-si7gptxw/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 2/ Creating a Vector Search Index on top of our Delta Lake table
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-3.1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC We now have our knowledge base ready, and saved as a Delta Lake table within Unity Catalog (including permission, lineage, audit logs and all UC features).
# MAGIC
# MAGIC Typically, deploying a production-grade Vector Search index on top of your knowledge base is a difficult task. You need to maintain a process to capture table changes, index the model, provide a security layer, and all sorts of advanced search capabilities.
# MAGIC
# MAGIC Databricks Vector Search removes those painpoints.
# MAGIC
# MAGIC ## Databricks Vector Search
# MAGIC Databricks Vector Search is a new production-grade service that allows you to store a vector representation of your data, including metadata. It will automatically sync with the source Delta table and keep your index up-to-date without you needing to worry about underlying pipelines or clusters. 
# MAGIC
# MAGIC It makes embeddings highly accessible. You can query the index with a simple API to return the most similar vectors, and can optionally include filters or keyword-based queries.
# MAGIC
# MAGIC Vector Search is currently in Private Preview; you can [*Request Access Here*](https://docs.google.com/forms/d/e/1FAIpQLSeeIPs41t1Ripkv2YnQkLgDCIzc_P6htZuUWviaUirY5P5vlw/viewform)
# MAGIC
# MAGIC If you still do not have access to Databricks Vector Search, you can leverage [Chroma](https://docs.trychroma.com/getting-started) (open-source embedding database for building LLM apps). For an example end-to-end implementation with Chroma, pleaase see [this demo](https://www.dbdemos.ai/minisite/llm-dolly-chatbot/). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Embeddings 
# MAGIC
# MAGIC The first step is to create embeddings from the documents saved in our Delta Lake table. To do so, we need an LLM model specialized in taking a text of arbitrary length, and turning it into an embedding (vector of fixed size representing our document). 
# MAGIC
# MAGIC Embedding creation is done through LLMs, and many options are available: from public APIs to private models fine-tuned on your datasets.
# MAGIC
# MAGIC *Note: It is critical to ensure that the model is always the same for both embedding index creation and real-time similarity search. Remember that if your embedding model changes, you'll have to re-index your entire set of vectors, otherwise similarity search won't return relevant results.*

# COMMAND ----------

# DBTITLE 1,Install vector search package
# MAGIC %pip install databricks-vectorsearch-preview git+https://github.com/mlflow/mlflow@master databricks-sdk==0.8.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=dbdemos $db=chatbot $reset_all_data=false

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating and registring our embedding model in UC
# MAGIC
# MAGIC Let's create an embedding model and save it in Unity Catalog. We'll then deploy it as serverless model serving endpoint. Vector Search will call this endpoint to create embeddings from our documents, and then index them.
# MAGIC
# MAGIC The model will also be used during realtime similarity search to convert the queries into vectors. This will be taken care of by Databricks Vector Search.
# MAGIC
# MAGIC #### Choosing an embeddings model
# MAGIC There are multiple choices for the embeddings model:
# MAGIC
# MAGIC * **SaaS API embeddings model**:
# MAGIC Starting simple with a SaaS API is a good option. If you want to avoid vendor dependency as a result of proprietary SaaS API solutions (e.g. OpenAI), you can build with a SaaS API that is pointing to an OSS model. You can use the new [MosaicML Embedding](https://docs.mosaicml.com/en/latest/inference.html) endpoint: `/instructor-large/v1`. See more in [this blogpost](https://www.databricks.com/blog/using-ai-gateway-llama2-rag-apps)
# MAGIC * **Deploy an OSS embeddings model**: On Databricks, you can deploy a custom copy of any OSS embeddings model behind a production-grade Model Serving endpoint.
# MAGIC * **Fine-tune an embeddings model**: On Databricks, you can use AutoML to fine-tune an embeddings model to your data. This has shown to improve relevance of retrieval. AutoML is in Private Preview - [Request Access Here](https://docs.google.com/forms/d/1MZuSBMIEVd88EkFj1ehN3c6Zr1OZfjOSvwGu0FpwWgo/edit)
# MAGIC
# MAGIC Because we want to keep this demo simple, we'll directly leverage MosaicML endpoint, an external SaaS API.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Deploying an AI gateway to MosaicML Endpoint
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-3.png?raw=true" style="float: right; margin-left: 10px"  width="600px;">
# MAGIC
# MAGIC
# MAGIC With MLflow, Databricks introduced the concept of AI Gateway ([documentation](https://mlflow.org/docs/latest/gateway/index.html)).
# MAGIC
# MAGIC AI Gateway acts as a proxy between your application and LLM APIs. It offers:
# MAGIC
# MAGIC - API key management
# MAGIC - Unified access point to easily switch the LLM backend without having to change your implementation
# MAGIC - Throughput control
# MAGIC - Logging and retries
# MAGIC - Format prompt for your underlying model
# MAGIC
# MAGIC *Note: if you don't have a MosaicML key, you can also deploy an OpenAI gateway route:*
# MAGIC
# MAGIC ```
# MAGIC gateway.create_route(
# MAGIC     name=mosaic_embeddings_route_name,
# MAGIC     route_type="llm/v1/embeddings",
# MAGIC     model={
# MAGIC         "name": "text-embedding-ada-002",
# MAGIC         "provider": "openai",
# MAGIC         "openai_config": {
# MAGIC             "openai_api_key": dbutils.secrets.get("dbdemos", "openai"),
# MAGIC         }
# MAGIC     }
# MAGIC )
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Creating the AI Gateway with MosaicML embedding 
#init MLflow experiment
import mlflow
from mlflow import gateway
init_experiment_for_batch("llm-chatbot-rag", "rag-model")

gateway.set_gateway_uri(gateway_uri="databricks")
#define our embedding route name, this is the endpoint we'll call for our embeddings
mosaic_embeddings_route_name = "mosaicml-instructor-xl-embeddings"

try:
    route = gateway.get_route(mosaic_embeddings_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_embeddings_route_name}")
    print(gateway.create_route(
        name=mosaic_embeddings_route_name,
        route_type="llm/v1/embeddings",
        model={
            "name": "instructor-xl",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="dbdemos", key="mosaic_ml_api_key")
            }
        }
    ))

# COMMAND ----------

# DBTITLE 1,Testing our AI Gateway
print(f"calling AI gateway {gateway.get_route(mosaic_embeddings_route_name).route_url}")

r = gateway.query(route=mosaic_embeddings_route_name, data={"text": "What is Databricks Lakehouse?"})

print(r)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Creating the Vector Search Index using our AI Gateway
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-4.png?raw=true" style="float: right; margin-left: 10px" width="600px">
# MAGIC
# MAGIC
# MAGIC Now that our embedding endpoint is up and running, we can use it in our Vector Search index definition.
# MAGIC
# MAGIC Every time a new row is added in our Delta Lake table, Databricks will automatically capture the change, call the embedding endpoint with the row content, and index the embedding.
# MAGIC
# MAGIC *Note: Databricks Vector Search can also use a custom endpoint if you wish to host your own fine-tuned embedding model*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Creating our Vector Search Catalog
# MAGIC As of now, Vector Search indexes live in a specific Catalog. Let's start by creating the catalog using the `VectorSearchClient`
# MAGIC
# MAGIC *Note: During the private preview, only 1 Vector Search catalog may be enabled and a few indexes may be defined.* 
# MAGIC

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClientV2
vsc = VectorSearchClientV2()

# COMMAND ----------

if not catalog_exists("vs_catalog"):
    print("creating Vector Search catalog")
    vsc.create_catalog("vs_catalog")
    #Make sure all users can access our VS index
    spark.sql(f"ALTER CATALOG vs_catalog SET OWNER TO `account users`")

#Wait for the catalog initialization which can take a few sec (see _resource/01-init for more details)
wait_for_vs_catalog_to_be_ready("vs_catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating our Index
# MAGIC
# MAGIC As reminder, we want to add the index in the `databricks_documentation` table, indexing the column `content`. Let's review our `databricks_documentation` table:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM databricks_documentation

# COMMAND ----------

# MAGIC %md
# MAGIC Vector search will capture all changes in your table, including updates and deletions, to synchronize your embedding index.
# MAGIC
# MAGIC To do so, make sure the `delta.enableChangeDataFeed` option is enabled in your Delta Lake table. Databricks Vector Index will use it to automatically propagate the changes. See the [Change Data Feed docs](https://docs.databricks.com/en/delta/delta-change-data-feed.html#enable-change-data-feed) to learn more, including how to set this property at the time of table creation

# COMMAND ----------

# MAGIC %sql 
# MAGIC ALTER TABLE databricks_documentation SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# DBTITLE 1,Creating the index
#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.databricks_documentation"
#Where we want to store our index
vs_index_fullname = f"vs_catalog.{db}.databricks_documentation_index"

#Use this to reset your catalog/index 
#vsc.delete_catalog("vs_catalog") #!!deleting the catalog will drop all the index!!
#vsc.delete_index(vs_index_fullname) #Uncomment to delete & re-create the index.
    
if not index_exists(vs_index_fullname):
    print(f'Creating a vector store index `{vs_index_fullname}` against the table `{source_table_fullname}`, using AI Gateway {mosaic_embeddings_route_name}')
    
    i = vsc.create_delta_sync_index(
        source_table_name=source_table_fullname,
        dest_index_name=vs_index_fullname,
        primary_key="id",
        column_to_embed="content",
        ai_gateway_route_name=mosaic_embeddings_route_name
    )
    sleep(3) #Set permission so that all users can access the demo index (shared)
    spark.sql(f'ALTER SCHEMA vs_catalog.{db} OWNER TO `account users`')
    set_index_permission(f"vs_catalog.{db}.databricks_documentation_index", "ALL_PRIVILEGES", "account users")
    print(i)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Waiting for the index to build
# MAGIC That's all we have to do. Under the hood, Databricks will maintain a [Delta Live Tables](https://docs.databricks.com/en/delta-live-tables/index.html) (DLT) job to refresh our pipeline.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, this can take several minutes.
# MAGIC
# MAGIC For more details, you can access the DLT pipeline from the link you get in the index definition.

# COMMAND ----------

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vs_index_fullname)
vsc.list_indexes("vs_catalog")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Your index is now ready and will be automatically synchronized with your table.
# MAGIC
# MAGIC Databricks will capture all changes made to the `databricks_documentation` Delta Lake table, and update the index accordingly. You can run your ingestion pipeline and update your documentations table, the index will automatically reflect these changes and get in synch with the best latencies.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Searching for similar content
# MAGIC
# MAGIC Our index is ready, and our Delta Lake table is now fully synchronized!
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also support a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call.*
# MAGIC
# MAGIC *Note: Make sure that what you search is similar to one of the documents you indexed! Check your document table if in doubt.*

# COMMAND ----------

question = "How can I track billing usage on my workspaces?"

results = vsc.get_index(vs_index_fullname).similarity_search(
  query_text=question,
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a Direct Index for realtime / API updates
# MAGIC
# MAGIC The index we created is synchronized with an existing Delta Lake table. To update the index, you'll have to update your Delta table. This works well for ~sec latencies workloads and Analytics needs.
# MAGIC
# MAGIC However, some use-cases requires sub-second update latencies. Databricks let you create real-time indexes using Direct Index. This let you run instant insert/update on your rows.
# MAGIC
# MAGIC ```
# MAGIC vsc.create_direct_vector_index(index_name=index_name,
# MAGIC   primary_key="id",
# MAGIC   embedding_dimension=1024,
# MAGIC   embedding_column="text_vector",
# MAGIC   schema={"id": "integer", "text": "string", "text_vector": "array<float>", "bool_val": "boolean"}
# MAGIC ```
# MAGIC   
# MAGIC You can then do instant updates: `vsc.get_index(index_name).upsert(new_rows)`

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your realtime chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [03-Deploy-RAG-Chatbot-Model]($./03-Deploy-RAG-Chatbot-Model) notebook to create and deploy a chatbot endpoint.
