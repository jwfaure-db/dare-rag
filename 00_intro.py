# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1: Ingest and prepare data
# MAGIC
# MAGIC To get started, you will set up the foundation of your generative AI application. You will focus on setting up a knowledge base by ingesting PDF documents. The knowledge base will be used for a later retrieval process.
# MAGIC
# MAGIC ## Detailed steps:
# MAGIC
# MAGIC - Upload PDFs to Amazon Simple Storage Service (S3) bucket
# MAGIC - Integrate Databricks and Amazon S3 by creating an [External Location](https://docs.databricks.com/en/sql/language-manual/sql-ref-external-locations.html)
# MAGIC - Set up [catalog](https://docs.databricks.com/en/connect/unity-catalog/managed-storage.html) and [volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html) within Databricks pointing to S3 directory containing PDFs
# MAGIC - Load PDFs into our `raw` table in binary format using [Auto Loader](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
# MAGIC - Parse text content from PDFs using unstructured [open-source library](https://github.com/Unstructured-IO/unstructured) within a Databricks User-Defined Function [(UDF)](https://docs.databricks.com/en/udf/index.html)
# MAGIC - Split text into chunks using [SentenceSplitter](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.SentenceSplitter.html) from [LlamaIndex](https://www.llamaindex.ai/)
# MAGIC - Process data into vectors by using an embeddings model
# MAGIC - Store text chunks + embeddings in a [Delta Lake](https://docs.databricks.com/en/delta/index.html) table
# MAGIC - Prepare for retrieval by similarity search using [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Setting up your AWS and Databricks environments
# MAGIC
# MAGIC Let's begin by setting up your Databricks and AWS environments.
# MAGIC
# MAGIC ## Setting up a compute cluster for your Databricks notebook
# MAGIC
# MAGIC Notebooks are a common tool in data science and machine learning for developing code and presenting results.
# MAGIC
# MAGIC To set up your notebook, you will need to first create a compute cluster.
# MAGIC
# MAGIC 1. Click **Compute** on the left side bar, then **Create compute**.
# MAGIC 2. Click **single node**, keep the default settings, then **Create compute**.
# MAGIC
# MAGIC ## Create External Location within Databricks
# MAGIC
# MAGIC Now, let's give your Databricks workspace the necessary permissions to use the data within your Amazon S3 bucket.
# MAGIC
# MAGIC > As part of your AWS account, a workshop-bucket-{AWS ACCOUNT ID} S3 bucket is pre-provisioned. To confirm, you can go to the Event Dashboard.
# MAGIC
# MAGIC To do this, we'll be creating an External Location.
# MAGIC
# MAGIC 1. Go to your Databricks Workspace Console, click **Catalog** to open Catalog Explorer.
# MAGIC 2. Click the **Add** button and select **Add an external location**.
# MAGIC 3. On the **Create a new external location** dialog, select **AWS Quickstart (Recommended)** and click **Next**.
# MAGIC 4. Enter the name of your S3 bucket (e.g `s3://workshop-bucket-{AWS ACCOUNT ID}`).
# MAGIC 5. Create a Personal Access Token, and Click **Launch in Quickstart**
# MAGIC
# MAGIC > The token provides access to Databricks APIs for the purposes of the workshop. In production, you can use service principals to provide greater security.
# MAGIC
# MAGIC 6. You will then be taken to the AWS console. Here, paste the Personal Access Token (PAT) you have created, and Click **Create Stack**. This will provision the resources necessary for this connection.
# MAGIC
# MAGIC > Note that the screenshot above refers to the AWS Console not Databricks.
# MAGIC
# MAGIC 7. Take a note of the CloudFormation stack name as this will be the location identifier that you will use in the next step. For example, `databricks-s3-ingest-ab123`.
# MAGIC
# MAGIC > Please be patient
# MAGIC > It may take up to 5 to 10 minutes for Databricks resources to be deployed in your AWS account.
# MAGIC
# MAGIC ## Create a new Catalog within Databricks
# MAGIC
# MAGIC Next, we will create a catalog, which is a container to organize your data assets such as schemas (databases), tables, and views.
# MAGIC
# MAGIC 1. Click **Catalog** on the left side bar, and click **Create catalog**.
# MAGIC 2. Create a unique name for your catalog `catalog_{AWS_ACCOUNT_ID}`.
# MAGIC 3. Choose the external location that you created earlier.
# MAGIC
# MAGIC > To filter for the correct S3 location, use the unique characters at the end of your CloudFormation stack name which you noted to create the external location. For example, search for `ab123` for the stack `databricks-s3-ingest-ab123`.
# MAGIC
# MAGIC 4. Specify the path as `databricks_cat`. This tells databricks to store catalog related data in the `databricks_cat` folder.
# MAGIC
# MAGIC > Warning: Do not use '`-`' characters in your catalog name, and use '`_`' instead.
# MAGIC
# MAGIC ## Create your own Databricks notebook
# MAGIC
# MAGIC We'll now create a new notebook to run the code for the workshop.
# MAGIC
# MAGIC 1. In the Databricks console, click on **workspace** and then **Home** within the left side bar. This will bring you to your own user directory within the workspace.
# MAGIC 2. Once you're in your own user directory, click on the **Add** button in the top right hand corner, and create a new notebook.
# MAGIC 3. Let's name this notebook `01-Data-Preparation`.
# MAGIC
# MAGIC ## Upload helper code and configurations into your Databricks Workspace
# MAGIC
# MAGIC Let's now load some helper code into our Databricks Workspace. The configuration and functions in the helper code will be referenced throughout your notebook activities in the workshop.
# MAGIC
# MAGIC 1. Download helper code to your local machine, by running the following command on your local machine.
# MAGIC    ```shell
# MAGIC    curl 'https://static.us-east-1.prod.workshops.aws/public/55b583a9-9503-4b0f-963c-7b6d7de428b9/assets/databricks_helper_code.zip'```
# MAGIC 2. Within your Databricks notebook, click on File in the top left hand corner, and then click on import notebook.
# MAGIC 3. Unzip helper_code.zip downloaded onto your local machine. Then, upload 3 files (00-init-advanced.py, 00-init.py, and config.py) into your Databricks user directory.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
