import streamlit as st


# Page Settings
no_indicator_calculation = st.Page("no_indicator_calculation.py", title="No Indicator Calculation", icon=":material/edit:")
with_indicator_calculation = st.Page("with_indicator_calculation.py", title="With Indicator Calculation", icon=":material/edit:")

pg = st.navigation([no_indicator_calculation, with_indicator_calculation])
st.set_page_config(
    page_title="Order Shaping Tool",
    page_icon=":milky_way:",
    layout="wide",
    menu_items={
        # "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "mailto:wang.n.22@pg.com",
        "About": """Developer: Wang Nan
                    \n Email: wang.n.22@pg.com"""
    }
)
pg.run()
