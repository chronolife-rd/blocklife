# %%
import os
import json

def get_env():
    DEV = os.environ.get('DEV', 'True')
    DEV = json.loads(DEV.lower())
    return DEV 