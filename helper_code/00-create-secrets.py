# Databricks notebook source
# Import the necessary libraries
import json
import requests

# COMMAND ----------

def create_scope(user_scope_name, access_token, workspace_url):
    # Set the API endpoint for creating a secret scope
    create_scope_endpoint = f"{workspace_url}/api/2.0/secrets/scopes/create"

    # Set the name of the secret scope you want to create
    scope_name = user_scope_name

    # Create the request payload
    payload = {
        "scope": scope_name,
    }

    # Set the request headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Make the API request to create the secret scope
    response = requests.post(create_scope_endpoint, data=json.dumps(payload), headers=headers)

    # Check the response status code
    if response.status_code == 200:
        print(f"Secret scope '{scope_name}' created successfully.")
    else:
        print(f"Failed to create secret scope. Status code: {response.status_code}")
        print(f"Error message: {response.text}")

# COMMAND ----------

def create_secret(key, value, user_scope_name, access_token, workspace_url):
    create_secret_endpoint = f"{workspace_url}/api/2.0/secrets/put"

    # Set the name of the secret scope and the secret key
    scope_name = user_scope_name
    secret_key = key
    # Set the secret value
    secret_value = value

    # Create the request payload
    payload = {
        "scope": scope_name,
        "key": secret_key,
        "string_value": secret_value
    }

    # Set the request headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    if key in [secret.key for secret in dbutils.secrets.list(scope_name)]:
        print("Secret already stored, replacing old secret")
    

    # Make the API request to create the secret
    response = requests.post(create_secret_endpoint, data=json.dumps(payload), headers=headers)

    # Check the response status code
    if response.status_code == 200:
        print(f"Secret '{secret_key}' created successfully in scope '{scope_name}'.")
    else:
        print(f"Failed to create secret. Status code: {response.status_code}")
        print(f"Error message: {response.text}")