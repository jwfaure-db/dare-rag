# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1: Ingest and Prepare Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Install dependencies and run helper code
# MAGIC
# MAGIC To get started, let's run the following code blocks within your new notebook. This installs dependencies, set environment variables, and run the helper code loaded into our Databricks Workspace. 
# MAGIC
# MAGIC *Please be patient: It may take up to 3-5 minutes for packages to install.*

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.5 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==2.5.2
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %run ./helper_code/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Storing your PDFs in binary format within Delta Tables
# MAGIC To get started with your RAG implementation, you will need to store your PDF data within Databricks. 
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/chatbot-rag/rag-pdf-1.png" style="display: block; margin: 0 auto"  width="900px;">
# MAGIC
# MAGIC First, let's: 
# MAGIC
# MAGIC * Import the necessary libaries you will require. Don't worry, you will learn about these libraries later on. 
# MAGIC * Configure the S3 location of your documents. This is based on the previously configured external location. 

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import io
import databricks.sdk.service.catalog as c
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from transformers import AutoTokenizer
from llama_index import Document, set_global_tokenizer
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
import re
from unstructured.partition.auto import partition

spark.conf.set("spark.sql.conf.bucket", S3_LOCATION + "/documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an external volume within your new Catalog
# MAGIC
# MAGIC Now let's create an [external volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html). An external volume is a Unity Catalog governed objects representing a logical volume of storage in a cloud object storage location, such as Amazon S3. You use volumes to store and access files in any format, including structured, semi-structured, and unstructured data.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE EXTERNAL VOLUME IF NOT EXISTS pdf_volume LOCATION '${spark.sql.conf.bucket}';

# COMMAND ----------

# MAGIC %md
# MAGIC Let's go ahead and upload PDFs into your external volume, then list the PDF documents:

# COMMAND ----------

volume_folder = f"/Volumes/{catalog}/{db}/pdf_volume"

if len(dbutils.fs.ls(volume_folder)) == 0:
    upload_pdfs_to_volume(volume_folder + "/llm_papers")

display(dbutils.fs.ls(volume_folder + "/llm_papers"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Storing PDFs in Binary Format
# MAGIC Now, let's ingest unstructured PDF data in [BINARY](https://docs.databricks.com/en/sql/language-manual/data-types/binary-type.html) type.
# MAGIC
# MAGIC Here, you use [Auto Loader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) to incrementally ingest new files. Auto Loader makes it easy to efficiently process billions of files from our S3 data lake into various data types.
# MAGIC
# MAGIC You then store this binary in a Delta table named `pdf_raw`. If you're not familiar, [Delta Lake](https://docs.databricks.com/en/delta/index.html)  is the optimized storage layer that provides the foundation for storing data and tables in the Databricks lakehouse.

# COMMAND ----------

df = (spark.readStream
      .format('cloudFiles')
      .option('cloudFiles.format', 'BINARYFILE')
      .load('dbfs:'+volume_folder + '/llm_papers'))

# Write the data as a Delta table
(df.writeStream
 .trigger(availableNow=True)
 .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/raw_docs')
 .table('pdf_raw').awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check if the documents have been written into our `pdf_raw` Delta Table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM pdf_raw LIMIT 2

# COMMAND ----------

# MAGIC %md
# MAGIC --- 
# MAGIC
# MAGIC ## 1.3 Processing your PDF Binary data
# MAGIC Great job! Your PDFs are now stored as BINARY within our `pdf_raw` Delta Table. The next step is to process them so that they are easily searchable. At a high-level, the below steps are required:
# MAGIC
# MAGIC * Extract text from PDFs using Optical Character Recognition (OCR).
# MAGIC * Breaking the text extracted into smaller chunks.
# MAGIC * Using an Embeddings Model to convert text into vectors.
# MAGIC * Store within a new `llm_pdf_documentation` Delta Table.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting text from PDFs
# MAGIC
# MAGIC First, you extract text from PDFs. This can be tricky as some PDFs are difficult to handle and may have been saved as images. Here, you use the [unstructured](https://github.com/Unstructured-IO/unstructured) open-source library within a [User-Defined Function](https://docs.databricks.com/en/udf/index.html) (UDF).
# MAGIC
# MAGIC To extract our PDF, let's install the relevant libraries in your notebook compute nodes. *Note: The `install_ocr_nodes()` function is defined in the helper code.*

# COMMAND ----------

install_ocr_on_nodes()

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's create the text extraction function for your PDFs:

# COMMAND ----------

def extract_doc_text(x: bytes) -> str:
    # Read files and extract the values with unstructured
    sections = partition(file=io.BytesIO(x))

    def clean_section(txt):
        txt = re.sub(r'\n', '', txt)
        return re.sub(r' ?\.', '.', txt)
    # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
    return "\n".join([clean_section(s.text) for s in sections])

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's test the text extraction function with a single PDF file:

# COMMAND ----------

with requests.get('https://github.com/databricks-demos/dbdemos-dataset/blob/main/llm/databricks-pdf-documentation/Databricks-Customer-360-ebook-Final.pdf?raw=true') as pdf:
    doc = extract_doc_text(pdf.content)
    print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC You should see that text content has been extracted from the raw [PDF file](https://github.com/databricks-demos/dbdemos-dataset/blob/main/llm/databricks-pdf-documentation/Databricks-Customer-360-ebook-Final.pdf?raw=true). This looks great, text data is now processed from the PDF!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunking
# MAGIC
# MAGIC In this example, some PDFs are very large, with a lot of text. You use [SentenceSplitter](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.SentenceSplitter.html) and ensure that each chunk isn't bigger than `500` tokens. 
# MAGIC
# MAGIC Remember that LLMs have a limited context window. As a general rule, it is recommended that Context + Instructions + Answer < Max Context Window.
# MAGIC
# MAGIC Let's create a function for us to do that chunking for us. We'll use [text_splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/) to split the long text into smaller chunks. You create a [pandas UDF](https://docs.databricks.com/en/udf/pandas.html) for high-performance processing.

# COMMAND ----------


spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # set llama2 as tokenizer to match our model size (will stay below BGE 1024 limit)
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    # Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

    def extract_and_split(b):
        txt = extract_doc_text(b)
        nodes = splitter.get_nodes_from_documents([Document(text=txt)])
        return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 1.4 Storing your data as vector embeddings within a Delta Table
# MAGIC The next step is to convert the chunked text into vectors using an Embeddings Model. This will allow for vector similarity search.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying your embeddings foundation model
# MAGIC The next step is to convert the chunked text into vectors using an Embeddings Model. This will allow for vector similarity search.
# MAGIC
# MAGIC [Databricks Model Serving](https://studio.us-east-1.prod.workshops.aws/preview/19ca57cd-5971-46c1-9c1b-e46afb6ca1ea/builds/55b583a9-9503-4b0f-963c-7b6d7de428b9/en-US/lab-1-ingest-your-data/04-embeddings#:~:text=Databricks%20Model%20Serving%C2%A0) supports serving a variety of models:
# MAGIC
# MAGIC - [Foundation Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) for access to models provided by Databricks.
# MAGIC - [External models](https://docs.databricks.com/en/generative-ai/external-models/index.html) which supports a variety of model providers, such as Amazon Bedrock .
# MAGIC - [Custom models](https://docs.databricks.com/en/machine-learning/model-serving/custom-models.html) which deploys a Python model as a production-grade API. These models can be trained using standard ML libraries like scikit-learn, XGBoost, PyTorch, and HuggingFace transformers and can include any Python code.
# MAGIC
# MAGIC In this example, we'll use the foundation models `BGE` provided by Databricks to convert our PDFs into vector embeddings. Let's see what this `BGE` model outputs for the query "What is Apache Spark?"

# COMMAND ----------

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api.
deploy_client = get_deploy_client("databricks")

embeddings = deploy_client.predict(
    endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
pprint(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Writing Vector Embeddings to Delta Table
# MAGIC Let's create a new Delta Table `llm_pdf_documentation` to store vector embeddings.

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS llm_pdf_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's create a [pandas UDF reference](<google.com>)  to process each chunk with your embeddings model to create its vector representation using the foundation model API.

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_embeddings(batch):
        # Note: this will fail if an exception is thrown during embedding creation (add try/except if needed)
        response = deploy_client.predict(
            endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size]
               for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's write some spark code to write to this new delta table `llm_pdf_documentation`.

# COMMAND ----------

(spark.readStream.table('pdf_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .selectExpr('path as url', 'content', 'embedding')
 .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_chunk')
    .table('llm_pdf_documentation').awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC To verify, check that the vector embeddings was successfully stored within our new Delta Table `llm_pdf_documentation`.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM llm_pdf_documentation

# COMMAND ----------

# MAGIC %md
# MAGIC Great job! You successfully processed data into its vector embeddings. Remember, these vector embeddings are simply numerical representation of data that makes it suitable for vector search. Let's continue.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Create Vector Search Index for your Delta Table
# MAGIC Let's create a Vector Search Index within Databricks to perform vector similarity search.
# MAGIC
# MAGIC Databricks provides multiple types of [Vector Search Index](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index):
# MAGIC - [Managed embeddings](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html#process-unstructured-data-and-databricks-managed-embeddings) where you provide a text column and endpoint name. Databricks synchronizes the index with your Delta Table.
# MAGIC - [Self-managed embeddings](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html#process-unstructured-data-and-customer-managed-embeddings) where you compute embeddings and store them in a field in a Delta Table, that can be synchronized with the index.
# MAGIC - **Direct vector search index** for real-time indexation use cases where you require the flexibility of no Delta Table and manage indexation using the API.
# MAGIC
# MAGIC In this example, you will learn how to setup a **self-managed embeddings index** on a Delta Table. To do so, you compute the embeddings of our chunks and save them in a Delta Table column as `array<float>`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Vector Search Endpoint
# MAGIC Previously, you transformed long documents into small chunks, computed embeddings and stored the vectors in a Delta Table.
# MAGIC
# MAGIC Next, you configure Databricks Vector Search to ingest data from this table. Databricks Vector Search provisions creates a vector search endpoint to serve the embeddings. In other words, you have a simple vector search API endpoint that you can use in your generative AI applications. Multiple indexes can use the same endpoint.
# MAGIC
# MAGIC Let's create our vector search endpoint:

# COMMAND ----------

vsc = VectorSearchClient(disable_notice=True)

if len(vsc.list_endpoints()) == 0 or VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)

print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC You can view your endpoint in your Databricks Workspace. Go to **Compute**. Then, click on the **Vector Search** tab. Then, click on the endpoint name to see all indexes served by the endpoint. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Creating Vector Search Index with self-managed embedding
# MAGIC
# MAGIC Let's create the self-managed vector search using our endpoint:

# COMMAND ----------

# The table we'd like to index
source_table_fullname = f"{catalog}.{db}.llm_pdf_documentation"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.llm_pdf_documentation_self_managed_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
    print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="TRIGGERED",  # Sync needs to be manually triggered
        primary_key="id",
        embedding_dimension=1024,  # Match your model embedding size
        embedding_vector_column="embedding"
    )
else:
    # Trigger a sync to update our vs content with the new data saved in the table
    print("Index exists, triggering sync between Delta Table and Vector Index")
    vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

# Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search for similar content
# MAGIC That's it! Databricks automatically captures and synchronizes new entries in your Delta Tables. Depending on your dataset and model size, it can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.

# COMMAND ----------

question = "How can I track billing usage on my workspaces?"

response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(query_vector=embeddings[0], columns=["url", "content"], num_results=1)

docs = results.get('result', {}).get('data_array', [])
pprint(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 [Optional] Using Amazon Titan Text Embeddings
# MAGIC Previously, you used BGE as an embeddings model. Let's explore the impact of using a different embeddings model. 
# MAGIC
# MAGIC Let's try an [external model](https://docs.databricks.com/en/generative-ai/external-models/index.html) `titan-embed-g1-text-02` from Amazon Bedrock. To deploy, you use AWS Credentials securely configured in [secret scopes](https://docs.databricks.com/en/security/secrets/secret-scopes.html).
# MAGIC

# COMMAND ----------

deploy_client = get_deploy_client('databricks')

if embeddings_model_endpoint_name not in [endpoints['name'] for endpoints in deploy_client.list_endpoints()]:
    deploy_client.create_endpoint(
        name=embeddings_model_endpoint_name,
        config={
            "served_entities": [
                {
                    "external_model": {
                        "name": "titan-embed-g1-text-02",
                        "provider": "amazon-bedrock",
                        "task": "llm/v1/embeddings",
                        "amazon_bedrock_config": {
                            "aws_region": "us-west-2",
                            "aws_access_key_id": "{{secrets/" + scope_name + "/aws_access_key_id}}",
                            "aws_secret_access_key": "{{secrets/" + scope_name + "/aws_secret_access_key}}",
                            "bedrock_provider": "amazon"
                        },
                    }
                }
            ]
        },
    )
else:
    print("external model for Titan Embeddings has already been created")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's compare Amazon Titan Text Embeddings vs. BGE.

# COMMAND ----------

bge_embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
titan_embeddings = deploy_client.predict(endpoint=embeddings_model_endpoint_name, inputs={"input": "What is Apache Spark?"})

# pprint(titan_embeddings)

print("titan embeddings model dimensions: " + str(len(titan_embeddings.get('data')[0].get('embedding'))))
print("bge embeddings model dimensions: " + str(len(bge_embeddings.get('data')[0].get('embedding'))))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that a key difference between Titan and BGE is the number of dimensions. In addition, there are implementation differences in the API input formats: BGE requires a list vs. Titan requires a string. 
# MAGIC
# MAGIC The choice of your embeddings model is crucial because it determines how you represent your data. This impacts your similarity search results.
# MAGIC
# MAGIC Let's continue with creating a Delta Table, Pandas UDF and Spark code for your Titan embeddings.

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS titan_llm_pdf_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

@pandas_udf("array<float>")
def get_titan_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_titan_embeddings(batch):
        # Note: this will fail if an exception is thrown during embedding creation (add try/except if needed)
        output = []
        for item in batch:
            response = deploy_client.predict(
                endpoint=embeddings_model_endpoint_name, inputs={"input": item})
            output.append(response.data[0]['embedding'])
        return output

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size]
               for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_titan_embeddings(batch.tolist())

    return pd.Series(all_embeddings)


# COMMAND ----------

(spark.readStream.table('pdf_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_titan_embedding("content"))
      .selectExpr('path as url', 'content', 'embedding')
 .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/titan_pdf_chunk')
    .table('titan_llm_pdf_documentation').awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC To verify, check that the vector embeddings was successfully stored within our new Delta Table `titan_llm_pdf_documentation`.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT embedding FROM titan_llm_pdf_documentation WHERE url like '%.pdf' limit 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a Vector Search Index using Amazon Titan Text Embeddings
# MAGIC
# MAGIC Let's create our vector search endpoint, along with the self-managed vector search using our endpoint. Note that multiple indexes can be deployed to the same endpoint.

# COMMAND ----------

# The table we'd like to index
source_table_fullname = f"{catalog}.{db}.titan_llm_pdf_documentation"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.titan_llm_pdf_documentation_self_managed_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
    print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="TRIGGERED",  # Sync needs to be manually triggered
        primary_key="id",
        embedding_dimension=1536,  # Match your model embedding size
        embedding_vector_column="embedding"
    )
else:
    # Trigger a sync to update our vs content with the new data saved in the table
    print("Index exists, triggering sync between Delta Table and Vector Index")
    vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

# Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's give it a try and search for similar content. Search results is dependent on the embeddings model you use.

# COMMAND ----------

question = "How can I track billing usage on my workspaces?"

response = deploy_client.predict(
    endpoint=embeddings_model_endpoint_name, inputs={"input": question})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
    query_vector=embeddings[0],
    columns=["url", "content"],
    num_results=1)
    
docs = results.get('result', {}).get('data_array', [])
pprint(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC In this Lab, you learned how Databricks simplifies and accelerates a foundational generative AI architecture. You learned how to ingest and prepare your documents. You then deployed a vector search endpoint with just a few lines of code. Next, you'll delve deeper into how you can leverage vector search to build an effective generative AI chatbot application.
# MAGIC
# MAGIC Next, open [02-Prepare-Model]($./02-Prepare-Model) to prepare your chat model.
