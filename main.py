import streamlit as st

from streamlit_option_menu import option_menu

import account, prediction,home,your
st.set_page_config(
    page_title="Predicted website",
)

class MultiApp:
    def __init__(self):
        self.apps = []
    def add_app(self,title,function):
        self.apps.append({
            "title":title,
            "function":function
        })
    def run():
        
        app = option_menu(
            menu_title='Main Menu',
            options=['account','prediction','home','MyPosts'],
            icons=['person-circle','house-fill','house-fill','chat-fill'],
            menu_icon='chat-text-fill',
            default_index=1,
            orientation="horizontal",
            styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )
        if app=='account':
            account.app()
        if app=='prediction':
            prediction.app()
        if app=='home':
            home.app()
        if app=='MyPosts':
            your.app()
        
    run()
                
        
    