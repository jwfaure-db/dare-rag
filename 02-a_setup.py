# Databricks notebook source
# MAGIC %md
# MAGIC # Setting up Databricks API access
# MAGIC
# MAGIC Let's begin by setting up your Databricks API access for your notebook. API access is required to access the vector search endpoint as set up in the previous lab.
# MAGIC
# MAGIC ## Creating your Databricks Personal Access Token (PAT)
# MAGIC
# MAGIC 1. In your Databricks Workspace, click your Databricks username in the top bar.
# MAGIC 2. Select **User Settings**, and click the **Developer** tab.
# MAGIC 3. Click **Manage** next to **Access Token** in the first line.
# MAGIC 4. Click **Generate New Token**.
# MAGIC
# MAGIC > Remember your token!
# MAGIC > Take a note of your Databricks Personal Access Token (PAT) as you will be using this token for multiple configurations. The token provides access to Databricks APIs for the purposes of the workshop. In production, you can use [service principals](#) to provide greater security.
# MAGIC
# MAGIC # Storing credentials securely
# MAGIC
# MAGIC Instead of directly entering your credentials into a notebook, you can use [Databricks secrets](#) to store your credentials and reference them.
# MAGIC
# MAGIC 1. Navigate to **Compute** on the left side bar, and click into the compute cluster that you have created for yourself.
# MAGIC 2. Click on **Apps** on the top navigation bar, and click into web terminal.
# MAGIC 3. Once you're in the web terminal, we'll configure our Databricks CLI using the following command.

# COMMAND ----------

curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
mv ~/bin/databricks /usr/local/sbin
databricks configure

# COMMAND ----------

# MAGIC %md
# MAGIC To authenticate, enter your Databricks Personal Access Token (PAT) that you noted above.
# MAGIC
# MAGIC 4. Let's define the name of our unique secret scope, and Databricks PAT generated earlier.
# MAGIC
# MAGIC > Remember to replace `{AWS_ACCOUNT_ID}` and `{DATABRICKS_PAT}` parameters. You can obtain your AWS account ID from `config.py`.
# MAGIC
# MAGIC 5. Next, we'll save the credential as `rag_sp_token` within the secret scope.
# MAGIC
# MAGIC

# COMMAND ----------

export DATABRICKS_PAT={DATABRICKS_PAT}
export SCOPE_NAME=scope_{AWS_ACCOUNT_ID}
databricks secrets create-scope $SCOPE_NAME
databricks secrets put-secret $SCOPE_NAME rag_sp_token --string-value $DATABRICKS_PAT

# COMMAND ----------

# MAGIC %md
# MAGIC 6. Now, navigate to `config.py` and verify that the variable `scope_name` is set to `scope_{AWS_ACCOUNT_ID}`.
# MAGIC
