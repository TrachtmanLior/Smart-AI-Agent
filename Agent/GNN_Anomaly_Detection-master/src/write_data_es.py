from elasticsearch import Elasticsearch, helpers
import pandas as pd
import time

# Elasticsearch connection
es = Elasticsearch("http://localhost:9200")
index_name = "network_traffic"

# Read CSV file
csv_file = "../data/elastic/malware_attack_11_11.csv"
df = pd.read_csv(csv_file)

# Convert DataFrame to list of dictionaries
records = df.to_dict(orient="records")

# Step 1: **Delete existing data in Elasticsearch**
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)  
    print(f"Deleted existing index: {index_name}")

# Step 2: **Recreate the index**
es.indices.create(index=index_name)
print(f"Created new index: {index_name}")

# Step 3: **Simulate real-time flow insertion**
batch_size = 500  
interval = 5  

print("Starting real-time data insertion...")

for i in range(0, len(records), batch_size):
    batch = records[i : i + batch_size]  

    # Convert batch to Elasticsearch bulk format
    actions = [
        {"_index": index_name, "_source": record}
        for record in batch
    ]

    # Insert batch into Elasticsearch
    helpers.bulk(es, actions)

    print(f"Inserted {len(batch)} flows into Elasticsearch")

    # Wait before inserting next batch
    time.sleep(interval)

time.sleep(30)
# Step 4: **Insert termination signal**
es.index(index=index_name, id="EOF", body={"message": "END_OF_FILE"})
print("Inserted termination signal into Elasticsearch.")

print("Finished inserting all flows!")


# from elasticsearch import Elasticsearch, helpers
# import pandas as pd
# import csv

# # Elasticsearch connection (adjust credentials if necessary)
# es = Elasticsearch("http://localhost:9200")  # Change to your ES host if needed
# index_name = "network_traffic"  # Change to desired index name

# # Read CSV file
# csv_file = "../data/elastic/malware_attack_11_11.csv"  # Change to your actual CSV file path
# df = pd.read_csv(csv_file)

# # Convert DataFrame to a list of dictionaries
# records = df.to_dict(orient="records")

# # Prepare data for Elasticsearch bulk insert
# def generate_data(records):
#     for record in records:
#         yield {
#             "_index": index_name,
#             "_source": record
#         }

# # Create the index (optional)
# if not es.indices.exists(index=index_name):
#     es.indices.create(index=index_name)

# # Insert data in bulk
# helpers.bulk(es, generate_data(records))

# print("Data inserted into Elasticsearch successfully!")
