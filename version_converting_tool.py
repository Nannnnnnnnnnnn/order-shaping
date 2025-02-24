import streamlit as st
from shareplum import Site
from shareplum import Office365
from shareplum.site import Version
import pandas as pd
import xlrd
import numpy as np
import gurobipy as grb
from st_aggrid import AgGrid, ColumnsAutoSizeMode
from io import BytesIO


# Page Settings
st.title("Version Converting Tool")
st.caption("Feel free to contact developer _Wang Nan_ if you have any question: wang.n.22@pg.com")
st.divider()


# Read Master Data
sharepointUsername = "wang.n.22@pg.com"
sharepointPassword = ""
sharepointSite = "https://pgone.sharepoint.com/sites/GCInnovationandCapabilityTeam"
website = "https://pgone.sharepoint.com"

authcookie = Office365(website, username=sharepointUsername, password=sharepointPassword).GetCookies()
site = Site(sharepointSite, version=Version.v365, authcookie=authcookie)
folder = site.Folder("Shared Documents/31. Order Loss Analysis/JD Full Truck Load/Order Shaping Tool/Input_MD")
shipto_city_data_filename = "JD B2C 线路明细.xlsx"

with open("shipto_city_data_temp.xlsx", mode='wb') as file:
    file.write(folder.get_file(shipto_city_data_filename))

shipto_city_data = pd.read_excel("shipto_city_data_temp.xlsx", dtype={"City": str, "品类": str, "shipto": str})


# File Uploading
uploaded_files = st.file_uploader(label="Please upload vertical/horizontal formatted order data file(s):", type=["xlsx", "xls"], accept_multiple_files=True)
city = list(shipto_city_data["Region"])

if len(uploaded_files) > 0:
    for uploaded_file in uploaded_files:
        # Data Initialization
        data = pd.read_excel(uploaded_file)
        if True in data.columns.str.contains("配送中心"):
            city_col_name = list(data.columns[data.columns.str.contains("配送中心")])[0]
            data["sku*"] = data["sku*"].map("{:.0f}".format)
            data = data.astype({"sku*": str, city_col_name: str})
            data = data.rename(columns={"sku*": "商品编码"})
            col_list = data.columns.tolist()
            keep_column_list = [col for col in col_list if
                                col not in ["采购需求数量*", city_col_name, "补货前周转天数", "补货后周转天数",
                                            "正负可调整周转天数"]]
            output_data = data.pivot(index=keep_column_list, columns=city_col_name, values=["采购需求数量*"])
            output_data.columns = [col[1] for col in output_data.columns.values]
            output_data = output_data.reset_index()
            suffix = "_横版.xlsx"
        else:
            data["商品编码"] = data["商品编码"].map("{:.0f}".format)
            data = data.astype({"商品编码": str})
            data = data.rename(columns={"商品编码": "sku*"})
            city_list = [col for col in data.columns.tolist() if col in city]
            var_list = [col for col in data.columns.tolist() if col not in city and col not in ["补货前周转天数", "补货后周转天数", "正负可调整周转天数"]]
            output_data = pd.melt(data, id_vars=var_list, value_vars=city_list, var_name="配送中心*(格式：北京,上海,广州)", value_name="采购需求数量*")
            suffix = "_竖版.xlsx"

        # File Downloading
        st.info("Convertion Result Download")
        output = BytesIO()
        uploaded_file_name = uploaded_file.name
        if uploaded_file_name.endswith(".xlsx"):
            file_name = uploaded_file_name.rstrip(".xlsx") + suffix
        elif uploaded_file_name.endswith(".xls"):
            file_name = uploaded_file_name.rstrip(".xls") + suffix
        else:
            file_name = "unknown_file_type_result.xlsx"
        with pd.ExcelWriter(output) as writer:
            output_data.to_excel(writer, index=False)
            st.download_button(
                label="Download Convertion Result for " + uploaded_file_name,
                data=output.getvalue(),
                file_name=file_name,
                mime="application/vnd.ms-excel"
            )
