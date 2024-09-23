import streamlit as st
import pandas as pd
import numpy as np
import gurobipy as grb
from st_aggrid import AgGrid, ColumnsAutoSizeMode

# Page Settings
st.set_page_config(
    page_title="Order Shaping Tool",
    page_icon=":milky_way:",
    layout="wide",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "mailto:wang.n.22@pg.com",
        'About': """Developer: Wang Nan
                    \n Email: wang.n.22@pg.com"""
    }
)
st.title("Order Shaping Tool")
st.caption("Feel free to contact developer _Wang Nan_ if you have any question: wang.n.22@pg.com")
st.divider()


# File Uploading
uploaded_file = st.file_uploader(label="Please upload the order and truck type data file:", type="xlsx")
if uploaded_file is not None:
    order_data = pd.read_excel(uploaded_file, sheet_name="Consolidated Input",
                               dtype={"Ship to": str, "配送中心名称": str, "品类名称": str, "商品名称": str, "宝洁八位码": str,
                                      "宝洁DC": str, "Sales Unit": str,
                                      "Min到货（箱数）": str, "Max到货（箱数）": str, "调整CS": str},
                               parse_dates=["预约到货时间"])
    truck_data = pd.read_excel(uploaded_file, dtype={"车型": str}, sheet_name="车型数据")


# Display Result
if uploaded_file is not None:
    st.info("Basic Info")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Ship-to")
            st.markdown("Customer")
            st.markdown("Size of Prize (RMB)")
            st.markdown("Quantity Changed (CS)")
        with col2:
            st.markdown("2002921387")
            st.markdown("北京京东世纪贸易有限公司")
            st.markdown("679")
            st.markdown("431")

    result = pd.DataFrame(
        {
            "Index": ["Unit Cost (RMB/PT)", "Loss (RMB)", "Truck Type", "VFR", "WFR", "Heavy/Light Mix(CBM/Ton)", "Quantity (CS)", "SLOG Impact"],
            "Before Shaping": ["90.1", "-971", "12.5*1+9.6GL*1+9.6*1", "74%", "93%", "2.5", "8710", "2.3%"],
            "After Shaping": ["78", "-292", "9.6GL*3", "87%", "93%", "2.5", "9141", "2.3%"],
            "Ideal State": ["72.8", "0", "9.6GL", "91%", "100%", "2.6", "N/A", "N/A"]
        }
    )

    result_grid_options = {
        "columnDefs": [
            {
                "headerName": "Index",
                "field": "Index",
                "cellStyle": {
                    "text-align": "center",
                    "fontWeight": "bold"
                },
                "sortable": False,
                "pinned": "left"
            },
            {
                "headerName": "Before Shaping",
                "field": "Before Shaping",
                "cellStyle": {
                    "text-align": "center"
                },
                "sortable": False
            },
            {
                "headerName": "After Shaping",
                "field": "After Shaping",
                "cellStyle": {
                    "text-align": "center"
                },
                "sortable": False
            },
            {
                "headerName": "Ideal State",
                "field": "Ideal State",
                "cellStyle": {
                    "text-align": "center"
                },
                "sortable": False
            }
        ]
    }

    custom_css = {
        ".ag-header-cell-label": {
            "justify-content": "center"
        }
    }

    st.info("Overall Result")
    AgGrid(
        result,
        gridOptions=result_grid_options,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        fit_columns_on_grid_load=True,
        theme="alpine",
        custom_css=custom_css
    )

    filler = pd.DataFrame(
        {
            "Category": order_data["品类名称"],
            "SKU Number": order_data["宝洁八位码"],
            "SKU Quantity Limit (-MIN CS, +MAX CS)": ["(-" + min_str + ", +" + max_str + ")" for min_str, max_str in zip(np.array(order_data["Min到货（箱数）"]), np.array(order_data["Max到货（箱数）"]))],
            "Reco Quantity (CS)": ["+" + cs_str for cs_str in order_data["调整CS"]]
        }
    )

    filler_grid_options = {
        "columnDefs": [
            {
                "headerName": "Category",
                "field": "Category",
                "filter": "agTextColumnFilter",
                # "suppressSizeToFit": True,
                "cellStyle": {
                    "text-align": "center"
                }
            },
            {
                "headerName": "SKU Number",
                "field": "SKU Number",
                "filter": "agTextColumnFilter",
                # "suppressSizeToFit": True,
                "cellStyle": {
                    "text-align": "center"
                }
            },
            {
                "headerName": "SKU Quantity Limit (-MIN CS, +MAX CS)",
                "field": "SKU Quantity Limit (-MIN CS, +MAX CS)",
                "filter": "agTextColumnFilter",
                # "suppressSizeToFit": True,
                "cellStyle": {
                    "text-align": "center"
                }
            },
            {
                "headerName": "Reco Quantity (CS)",
                "field": "Reco Quantity (CS)",
                "filter": "agNumberColumnFilter",
                # "suppressSizeToFit": True,
                "cellStyle": {
                    "text-align": "center"
                }
            }
        ]
    }

    st.info("Filler SKU Recommendation")
    AgGrid(
        filler,
        gridOptions=filler_grid_options,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        fit_columns_on_grid_load=True,
        theme="alpine",
        custom_css=custom_css
    )
