import streamlit as st 
from constant import (init_session_state, 
                      restart_session_state,
                      URL_ROOT,
                      URL_V1,
                      LOGO_CLIFE,
                      )
from back_functions import (api_auth, 
                            test_string,
                            api_auth_v1,
                            )
from template.style import hide_menu, set_footer_style
from version import VERSION
import requests 
import json
import pandas as pd
from back_functions import get_users
from back_functions import get_status_table
from back_functions import update_status

st.set_page_config("Block life",layout="wide",)
# hide_menu()
init_session_state()

# %%
with st.sidebar:
    col_logo,_=st.columns(2)
    col_logo.image(LOGO_CLIFE)
    
    username        = st.text_input("Username", st.session_state['username'], placeholder="Ex: Chronnolife")
    api_key         = st.text_input("API key", st.session_state['api_key'], placeholder="Ex: f9VBqQoTiU0mnAKoXK1lky", type="password")
    api_key_v1      = st.text_input("Second API key", st.session_state['api_key_v1'], placeholder="Ex: f9VBqQoTiU0mnAKoXK1lky", type="password")
    
    if api_key:
        st.session_state.api_key = api_key
        
    if api_key_v1:
        st.session_state.api_key_v1 = api_key_v1

    col1, col2 = st.columns(2)
    button_sign_in = col1.button('Connect')
    button_sign_out = col2.button('Change account')
    
    if button_sign_in:
        
        username, error         = test_string(username, name="username", layout=st.sidebar)
        api_key, error          = test_string(api_key, name="API key", layout=st.sidebar)
        api_key_v1, error       = test_string(api_key_v1, name="Second API key", layout=st.sidebar)
        
        if not error:
            st.session_state.username   = username
            st.session_state.api_key    = api_key
            st.session_state.api_key_v1 = api_key_v1
        
        # % GET: Retrieve relevant properties of the specified user.
        message, status_code = api_auth()
        
        # % POST. Login with token
        message_v1, status_code_v1 = api_auth_v1()
        
        if status_code == 200 and status_code_v1 == 200 :
            st.session_state.auth_status = True
        else:
            st.session_state.auth_status = False
            
        if st.session_state.auth_status:
            st.success(message)
        else:
            if status_code != 200:
                st.error(message)
            if status_code_v1 != 200:
                st.error(message_v1)
            
    if button_sign_out:
       st.session_state.auth_status = False 
       restart_session_state()
       st.warning('Not connected')

with st.container():
    if st.session_state.auth_status:
        
        form_search = st.form('form_search')
        client_username = form_search.text_input("Client username")
        search = form_search.form_submit_button("Display endusers activation status")
        
        if search:
            client_username, error  = test_string(client_username, name="client username", layout=st.sidebar)
            st.session_state.client_username = client_username
            error = get_users()
                    
            if len(error) > 0:
                st.error(error)
                st.stop()
                
            if len(st.session_state.users) == 0:
                st.warning("No end users found")
                st.stop()
            
            with st.spinner("Searching for endusers..."):
                get_status_table()
            
        df = st.session_state.df
        if df is not None:
            layout = st.empty()
            form = layout.form('form')
            edited_df = form.experimental_data_editor(df, height=int((len(df)+1)*35))
            button_apply = form.form_submit_button('Apply')
        
            if button_apply:
                count = update_status(edited_df)
                if count > 0:
                    layout.empty()
                else:
                    st.info("Change Blocked status and click on Apply again")
            
    else:
        st.info('Please enter a Username, an API-key and a Second API-key to access Block life')
        
        
footer = set_footer_style()
footer +="""
<div class="footer">
<p style="color: grey;">Version
"""
footer += VERSION 
footer +="""
</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

