# test_rsa.py
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import snowflake.connector

with open('snowflake_key.p8', 'rb') as f:
    key = serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

key_bytes = key.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

conn = snowflake.connector.connect(
    account='SFEDU02-FEB92475',
    user='BADGER',
    private_key=key_bytes,
    database='ECOMMERCE_DW',
    warehouse='ECOMMERCE_LOAD_WH',
    role='TRAINING_ROLE'
)

cursor = conn.cursor()
cursor.execute("SELECT CURRENT_USER(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
print(cursor.fetchone())
cursor.close()
conn.close()
print("Connection successful!")