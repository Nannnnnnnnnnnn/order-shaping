import streamlit as st


# Page Settings
no_indicator_calculation = st.Page("no_indicator_calculation.py", title="No Indicator Calculation", icon=":material/edit:")
with_indicator_calculation = st.Page("with_indicator_calculation.py", title="With Indicator Calculation", icon=":material/edit:")

pg = st.navigation([no_indicator_calculation, with_indicator_calculation])
st.set_page_config(page_title="Order Shaping Tool", page_icon=":milky_way:")
pg.run()
