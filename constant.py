import streamlit as st
import os
from PIL import Image

URL_V1      = "https://prod.chronolife.net/api/1/user/events"
URL_ROOT    = "https://prod.chronolife.net/api/2"
URL_DATA    = URL_ROOT + "/data"

CURRENT_DIRECTORY = os.getcwd()

LOGO_CLIFE = Image.open(CURRENT_DIRECTORY + '/assets/logoclife.png')


def init_session_state():
    if 'username' not in st.session_state:
        st.session_state.username = ''
        
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
        
    if 'api_key_v1' not in st.session_state:
        st.session_state.api_key_v1 = ''

    if 'client_username' not in st.session_state:
            st.session_state.client_username = ''
            
    if 'users' not in st.session_state:
            st.session_state.users = ''

    if 'auth_status' not in st.session_state:
            st.session_state.auth_status = False

    if 'df' not in st.session_state:
            st.session_state.df = None
        
def restart_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
    init_session_state()
    st.experimental_rerun()
    