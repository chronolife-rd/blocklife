import requests
from constant import URL_ROOT
from constant import URL_V1
import streamlit as st
import json
import pandas as pd
import time

def api_auth():
    status_code = None
    message = None
    
    username = st.session_state.username
    api_key = st.session_state.api_key
    url = URL_ROOT + "/user/{userId}".format(userId=username)
    
    reply = requests.get(url, headers={"X-API-Key": api_key})
    status_code = reply.status_code
    message = api_status(reply.status_code)

    return message, status_code

def api_auth_v1():
    status_code = None
    message = None
    
    username = st.session_state.username
    api_key_v1 = st.session_state.api_key_v1
    json_request = []
    r = requests.post(URL_V1, auth=(username, api_key_v1), json=json_request)
    message = api_status(r.status_code, api_version=1)
    status_code = r.status_code
    
    return message, status_code

def api_status(status_code, user_text='username', api_version=2):
    
    message = ""
    
    span = ""
    if api_version==1:
        span = "Second "
    
    if status_code == 200:
        message = 'Connected'
    elif status_code == 400:
        message = 'Part of the request could not be parsed or is incorrect'
    elif status_code == 401:
        message = 'Incorrect ' + span + 'API key'
    elif status_code == 403:
        message = 'Not authorized'
    elif status_code == 404:
        message = 'Incorrect url'
    elif status_code == 500:
        message = 'Incorrect ' + user_text
    elif status_code == 0:
        message = "You are disconnect"
        
    return message

def test_string(string, name, layout):
    
    error = False
    message = False
    if len(string) > 0:
        string = string.replace(" ", "")
    else:
        message = "Please fill in " + name + ' field'
        layout.error(message)
        error = True
        return string, error
        
    return string, error
    
def get_subusers():
    client_username = st.session_state.client_username
    api_key         = st.session_state.api_key
    url             = URL_ROOT + "/user/{userId}".format(userId=client_username)
    reply           = requests.get(url, headers={"X-API-Key": api_key})
    
    error = ''
    users = []
    
    if reply.status_code == 200:
        users = json.loads(reply.text)['users'] 
    elif reply.status_code == 400:
        error = 'Part of the request could not be parsed or is incorrect.'
    elif reply.status_code == 401:
        error = 'Invalid authentication'
    elif reply.status_code == 403:
        error = 'Not authorized.'
    elif reply.status_code == 404:
        error = 'Invalid url'
    elif reply.status_code == 500:
        error = 'Invalid user ID'
        
    st.session_state.users = users
    return error

def get_status_table():
    users = st.session_state.users
    api_key = st.session_state.api_key
    dict_table = []
    for user in users:
        try: 
            if isinstance(int(user[0]), int):
                
                url = URL_ROOT + "/user/{userId}/blocked".format(userId=user)
                request_body = {'blocked': False}
                reply = requests.put(url, headers={"X-API-Key": api_key}, json=request_body)
                json_list_of_records = []
                if reply.status_code == 200:
                    json_list_of_records = json.loads(reply.text) 
                else:
                    st.error('Request failed')
                    st.stop()
                time.sleep(0.1)
                if json_list_of_records["nb_updated_user"] > 0:
                    request_body = {'blocked': True}
                    reply = requests.put(url, headers={"X-API-Key": api_key}, json=request_body)
                    blocked = True
                else:
                    blocked = False
                    
                dict_table.append({"Enduser": user, "Blocked": blocked})
        except:
            pass
        
    df = pd.DataFrame(dict_table)
    st.session_state.df = df
    
def update_status(edited_df):   
    api_key = st.session_state.api_key
    df = st.session_state.df
    status_s = []
    count = 0
    for index, row in edited_df.iterrows():
        if row["Blocked"] != df["Blocked"].iloc[index]:
            if row["Blocked"]: 
                request_body = {'blocked': True}
                status = "Blocked"
            else:
                request_body = {'blocked': False}
                status = "Activated"
            status_s.append(status)
            url = URL_ROOT + "/user/{userId}/blocked".format(userId=row["Enduser"])
            reply = requests.put(url, headers={"X-API-Key": api_key}, json=request_body)
            time.sleep(0.1)
            if reply.status_code != 200:
                st.error('Request failed')
                st.stop()
            else:
                st.success((row["Enduser"] + " has been " + status))
                
            count+=1
            
    return count