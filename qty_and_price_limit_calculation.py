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
st.title("Order Shaping Tool")
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
truck_master_filename = "Tariff.xlsx"
ideal_truck_type_master_filename = "Ideal Truck Type.xlsx"
sku_transfer_data_filename = "产品主数据_20241015030419.xlsx"
sku_price_data_filename = "京东直供数据2024_1010-非美妆.xlsx"
shipto_city_data_filename = "JD B2C 线路明细.xlsx"
sku_master_filename = "SKU主数据.xlsx"

with open("truck_master_temp.xlsx", mode='wb') as file:
    file.write(folder.get_file(truck_master_filename))

truck_master = pd.read_excel("truck_master_temp.xlsx", dtype={"Ship-to": str, "Truck Type": str, "Optimal Truck Type": str})

with open("ideal_truck_type_master_temp.xlsx", mode='wb') as file:
    file.write(folder.get_file(ideal_truck_type_master_filename))

ideal_truck_type_master = pd.read_excel("ideal_truck_type_master_temp.xlsx", dtype={"Ship-to": str, "Customer Name": str})

with open("sku_transfer_data_temp.xlsx", mode='wb') as file:
    file.write(folder.get_file(sku_transfer_data_filename))

sku_transfer_data = pd.read_excel("sku_transfer_data_temp.xlsx", dtype={"京东码": str, "宝洁码": str, "箱规": int})

with open("sku_price_data_temp.xlsx", mode='wb') as file:
    file.write(folder.get_file(sku_price_data_filename))

sku_price_data = pd.read_excel("sku_price_data_temp.xlsx", dtype={"京东码": str, "宝洁码": str, "箱规": int})

with open("shipto_city_data_temp.xlsx", mode='wb') as file:
    file.write(folder.get_file(shipto_city_data_filename))

shipto_city_data = pd.read_excel("shipto_city_data_temp.xlsx", dtype={"City": str, "品类": str, "shipto": str})

with open("sku_master_temp.xlsx", mode='wb') as file:
    file.write(folder.get_file(sku_master_filename))

sku_master = pd.read_excel("sku_master_temp.xlsx", dtype={"material_num": str, "category": str})


# File Uploading
uploaded_files = st.file_uploader(label="Please upload the order and truck type data file:", type=["xlsx", "xls"], accept_multiple_files=True)
original_ao_order_data = pd.DataFrame()
order_data = pd.DataFrame()
column_dict = {}
selected_shipto = ["2003241647", "2002921387"]
selected_category = ["FHC", "HC", "PCC"]
city = list(shipto_city_data["Region"])
exist_order_flag = "N"
adopt_calculation_flag = "N"


# Data Initialization
if len(uploaded_files) > 0:
    for uploaded_file in uploaded_files:
        data_split = pd.read_excel(uploaded_file)
        if True in data_split.columns.str.contains("POA回告数量"):
            original_ao_order_data_split = data_split.astype({"Ship to编码": str, "宝洁八位码": str})
            original_ao_order_data_split = original_ao_order_data_split.rename(columns={"Ship to编码": "shipto", "宝洁八位码": "material_num", "POA回告数量（箱数）": "CS", "POA回告数量（件数）": "IT"})
            original_ao_order_data_split = pd.merge(original_ao_order_data_split, sku_master.loc[:, ["material_num", "category", "volume_cube", "weight_ton"]], how="left", on="material_num")
            original_ao_order_data_split = pd.merge(original_ao_order_data_split , sku_price_data.loc[:, ["宝洁码", "仓报价"]], how="left", left_on="material_num", right_on="宝洁码")
            original_ao_order_data_split["箱规"] = original_ao_order_data_split["IT"] / original_ao_order_data_split["CS"]
            original_ao_order_data_split["仓报价"] = original_ao_order_data_split["仓报价"] * original_ao_order_data_split["箱规"]
            original_ao_order_data = pd.concat([original_ao_order_data, original_ao_order_data_split])
        else:
            exist_order_flag = "Y"
            column_dict[uploaded_file.name] = data_split.columns.tolist()
            city_col_name = list(data_split.columns[data_split.columns.str.contains("配送中心")])[0]
            data_split["sku*"] = data_split["sku*"].map("{:.0f}".format)
            original_order_data_split = data_split.astype({"sku*": str, city_col_name: str, "补货前周转天数": float, "补货后周转天数": float, "正负可调整周转天数": float})
            
            order_data_split = original_order_data_split.rename(columns={"sku*": "京东码", city_col_name: "配送中心*(格式：北京,上海,广州)"})
            order_data_split["Source"] = uploaded_file.name + "_竖版"
            if True in order_data_split.columns.str.contains("实际采纳数量"):
                adopt_qty_col_name = list(order_data_split.columns[order_data_split.columns.str.contains("实际采纳数量")])[0]
                if not order_data_split[adopt_qty_col_name].isnull().any():
                    adopt_calculation_flag = "Y"
                    order_data_split["adopt_qty"] = order_data_split[adopt_qty_col_name].fillna(0)
                else:
                    order_data_split["adopt_qty"] = order_data_split["采购需求数量*"]
            else:
                order_data_split["adopt_qty"] = order_data_split["采购需求数量*"]
            sku_data = pd.merge(sku_transfer_data.loc[:, ["京东码", "宝洁码", "箱规"]], sku_master.loc[:, ["material_num", "category", "volume_cube", "weight_ton"]], how="outer", left_on="宝洁码", right_on="material_num")
            sku_data = sku_data[(sku_data["京东码"].notnull()) & (sku_data["material_num"].notnull())]
            sku_data.drop_duplicates(subset=["京东码"], inplace=True)
            order_data_split = pd.merge(order_data_split, sku_data.loc[:, ["京东码", "宝洁码", "箱规", "category", "volume_cube", "weight_ton"]], how="left", on="京东码")
            order_data_split = pd.merge(order_data_split, sku_price_data.loc[:, ["京东码", "仓报价"]], how="left", on="京东码")
            order_data_split["仓报价"] = order_data_split["仓报价"] * order_data_split["箱规"]
            order_data_split["CS"] = order_data_split["采购需求数量*"] / order_data_split["箱规"]
            order_data_split["max_filler_CS"] = order_data_split["CS"] / (order_data_split["补货后周转天数"] - order_data_split["补货前周转天数"]) * order_data_split["正负可调整周转天数"]
            order_data_split["max_filler_CS"] = order_data_split[["CS", "max_filler_CS"]].min(axis=1)
            order_data_split = order_data_split.rename(columns={"宝洁码": "material_num"})
            order_data_split["Region"] = order_data_split["配送中心*(格式：北京,上海,广州)"]
            if any(uploaded_file.name.startswith(file_category := category) for category in selected_category):
                order_data_split["category"] = file_category
                source_shipto_city_data = shipto_city_data[shipto_city_data["品类"].str.startswith(file_category + "/", na=False) | shipto_city_data["品类"].str.contains("/" + file_category, na=False)]
                order_data_split = pd.merge(order_data_split, source_shipto_city_data.loc[:, ["Region", "shipto"]], how="left", on="Region")
            else:
                st.warning("**Warning:**" + " The name(s) of the uploaded order data file(s) should **_start(s) with_** the category short name, e.g. \"FHC-周转单-8.26-168w-横版\". Please check the name(s) and upload the file(s) again.")
                st.stop()
            order_data = pd.concat([order_data, order_data_split])
    if exist_order_flag == "Y":
        upload_source_list = list(set(list(order_data["Source"])))
        not_selected_shipto = list(set(list(order_data["shipto"])) - set(selected_shipto))
        not_selected_order_data = order_data[(order_data["shipto"].isin(not_selected_shipto)) | (order_data["采购需求数量*"] == 0)]
        selected_order_data = order_data[(order_data["shipto"].isin(selected_shipto)) & (order_data["采购需求数量*"] != 0)]
        download_source_list = list(set(list(selected_order_data["Source"])))
        missing_sku_material_list = ", ".join("{0}".format(sku) for sku in list(selected_order_data[selected_order_data["material_num"].isnull()]["京东码"]))
        missing_sku_info_list = ", ".join("{0}".format(sku) for sku in list(selected_order_data[(selected_order_data["material_num"].notnull()) & ((selected_order_data["volume_cube"].isnull()) | (selected_order_data["weight_ton"].isnull()))]["京东码"]))
        missing_sku_price_list = ", ".join("{0}".format(sku) for sku in list(selected_order_data[(selected_order_data["material_num"].notnull()) & (selected_order_data["仓报价"].isnull())]["京东码"]))
        if len(missing_sku_material_list) > 0:
            st.warning("**Warning:**" + " Missing material number master data of SKU " + missing_sku_material_list + " for the order data, which will be excluded from the optimization.")
        if len(missing_sku_info_list) > 0:
            st.warning("**Warning:**" + " Missing category/weight/volume master data of SKU " + missing_sku_info_list + " for the order data, which will be excluded from the optimization.")
        if len(missing_sku_price_list) > 0:
            st.warning("**Warning:**" + " Missing price master data of SKU " + missing_sku_price_list + " for the order data, which will be excluded from the optimization.")
        if len(original_ao_order_data) > 0:
            not_selected_ao_order_data = original_ao_order_data[(original_ao_order_data["shipto"].isin(not_selected_shipto)) & (original_ao_order_data["CS"] == 0)]
            selected_ao_order_data = original_ao_order_data[(original_ao_order_data["shipto"].isin(selected_shipto)) & (original_ao_order_data["CS"] != 0)]
            missing_ao_sku_info_list = ", ".join("{0}".format(sku) for sku in list(selected_ao_order_data[(selected_ao_order_data["category"].isnull()) | (selected_ao_order_data["volume_cube"].isnull()) | (selected_ao_order_data["weight_ton"].isnull())]["material_num"]))
            missing_ao_sku_price_list = ", ".join("{0}".format(sku) for sku in list(selected_ao_order_data[(selected_ao_order_data["仓报价"].isnull())]["material_num"]))
            if len(missing_ao_sku_info_list) > 0:
                st.warning("**Warning:**" + " Missing category/weight/volume master data of SKU " + missing_ao_sku_info_list + " for the AO order data, which will be excluded from the optimization.")
            if len(missing_ao_sku_price_list) > 0:
                st.warning("**Warning:**" + " Missing price master data of SKU " + missing_ao_sku_price_list + " for the AO order data, which will be excluded from the optimization.")
    else:
        st.warning("**Warning:**" + " No order data uploaded (excluding AO order data), the optimization terminated.")


# Calculation Function
def baseline(initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume, truck_cost):
    # Construct Model Object
    model = grb.Model("baseline")

    # Introduce Decision Variables & Parameters
    n = len(truck_capacity_weight)
    x = [[] for i in range(n)]
    for i in range(n):
        x[i] = model.addVar(lb=0, ub=float("inf"), vtype=grb.GRB.INTEGER, name="x_" + str(i))

    # Objective Function
    cost = grb.LinExpr(truck_cost, x)
    model.setObjective(cost, grb.GRB.MINIMIZE)

    # Constraints
    model.addConstr(grb.LinExpr(truck_capacity_weight, x) >= initial_order_weight, "Capacity Constraint")
    model.addConstr(grb.LinExpr(truck_capacity_volume, x) >= initial_order_volume, "Capacity Constraint")

    # Solve the Constructed Model
    model.optimize()
    for var in model.getVars():
        if model.status == grb.GRB.OPTIMAL:
            print(var.varName, "\t", var.x)

    total_capacity_weight = 0
    total_capacity_volume = 0
    truck_loading_weight = initial_order_weight
    truck_loading_volume = initial_order_volume
    truck_qty = []

    if model.status == grb.GRB.OPTIMAL:
        for var in model.getVars():
            truck_qty.append(var.x)
            total_capacity_weight += var.x * truck_capacity_weight[var.index]
            total_capacity_volume += var.x * truck_capacity_volume[var.index]

    cost = model.objVal
    unit_cost = cost / max(truck_loading_weight, truck_loading_volume / 3)
    pt = max(truck_loading_weight, truck_loading_volume / 3)
    wfr = truck_loading_weight / total_capacity_weight
    vfr = truck_loading_volume / total_capacity_volume
    mix = truck_loading_volume / truck_loading_weight

    return truck_qty, cost, unit_cost, pt, wfr, vfr, mix


def model_execution(initial_order_weight, initial_order_volume, order_unit_weight,
                    order_unit_volume, order_unit_price, category_key, category_index_lb, category_index_ub, category_price, max_qty, min_qty, priority_param, truck_capacity_weight, truck_capacity_volume,
                    truck_cost):
    # Construct Model Object
    model = grb.Model("JD_Order")

    # Parameters
    M = 100000

    max_pt = max(initial_order_weight + sum(np.multiply(order_unit_weight, max_qty)),
                 (initial_order_volume + sum(np.multiply(order_unit_volume, max_qty))) / 3)
    min_pt = max(initial_order_weight + sum(np.multiply(order_unit_weight, min_qty)),
                 (initial_order_volume + sum(np.multiply(order_unit_volume, min_qty))) / 3)

    # Introduce Decision Variables & Parameters
    n = len(order_unit_weight)
    m = len(truck_capacity_weight)
    l = n * 2 + m * 2 + 2
    p = model.addVar(lb=1 / max_pt, ub=1 / min_pt, vtype=grb.GRB.CONTINUOUS, name="p")
    q = [[] for i in range(l)]
    for i in range(l):
        if i in range(n):
            if min_qty[i] >= 0:
                q[i] = model.addVar(lb=min_qty[i] / max_pt, ub=max_qty[i] / min_pt, vtype=grb.GRB.CONTINUOUS, name="q_" + str(i))
            else:
                q[i] = model.addVar(lb=min_qty[i] / min_pt, ub=max_qty[i] / min_pt, vtype=grb.GRB.CONTINUOUS, name="q_" + str(i))
        elif i in range(n, n + m):
            q[i] = model.addVar(lb=0, ub=float("inf"), vtype=grb.GRB.CONTINUOUS, name="q_" + str(i))
        elif i in range(n + m, n * 2 + m):
            q[i] = model.addVar(lb=min_qty[i - n - m], ub=max_qty[i - n - m], vtype=grb.GRB.INTEGER, name="q_" + str(i))
        elif i in range(n * 2 + m, n * 2 + m * 2):
            q[i] = model.addVar(lb=0, ub=float("inf"), vtype=grb.GRB.INTEGER, name="q_" + str(i))
        else:
            q[i] = model.addVar(vtype=grb.GRB.BINARY, name="q_" + str(i))

    # Objective Function
    cost_per_pt = grb.LinExpr(truck_cost, q[n:(n + m)])
    priority = grb.LinExpr(priority_param, q[:n])
    model.setObjectiveN(cost_per_pt, index=0, priority=10, weight=1, name="Cost/PT")
    model.setObjectiveN(priority, index=4, priority=0, name="Priority")

    # Constraints
    model.addConstr(grb.LinExpr(order_unit_weight, q[:n]) + initial_order_weight * p <=
                    grb.LinExpr(truck_capacity_weight, q[n:(n + m)]), "Capacity Constraint")
    model.addConstr(grb.LinExpr(order_unit_volume, q[:n]) + initial_order_volume * p <=
                    grb.LinExpr(truck_capacity_volume, q[n:(n + m)]), "Capacity Constraint")
    model.addConstr(grb.LinExpr(order_unit_weight, q[:n]) + initial_order_weight * p <= 1, "PT Constraint")
    model.addConstr((grb.LinExpr(order_unit_volume, q[:n]) + initial_order_volume * p) / 3 <= 1, "PT Constraint")
    model.addConstr(grb.LinExpr(order_unit_weight, q[:n]) + initial_order_weight * p >=
                    1 - M * (1 - q[n * 2 + m * 2]), "PT Constraint")
    model.addConstr((grb.LinExpr(order_unit_volume, q[:n]) + initial_order_volume * p) / 3 >=
                    1 - M * (1 - q[n * 2 + m * 2 + 1]), "PT Constraint")
    model.addConstr(q[n * 2 + m * 2] + q[n * 2 + m * 2 + 1] >= 1, "PT Constraint")
    for index in range(len(category_key)):
        lb = category_index_lb[index]
        ub = category_index_ub[index]
        initial_order_price = category_price[index]
        if ub != n - 1:
            ub += 1
            model.addConstr(grb.LinExpr(order_unit_price[lb:ub], q[lb:ub]) <= initial_order_price * p * 0.05,
                            "Category Price Constraint")
            model.addConstr(grb.LinExpr(order_unit_price[lb:ub], q[lb:ub]) >= - initial_order_price * p * 0.05,
                            "Category Price Constraint")
        else:
            model.addConstr(grb.LinExpr(order_unit_price[lb:], q[lb:n]) <= initial_order_price * p * 0.05,
                            "Category Price Constraint")
            model.addConstr(grb.LinExpr(order_unit_price[lb:], q[lb:n]) >= - initial_order_price * p * 0.05,
                            "Category Price Constraint")
    for i in range(n):
        model.addQConstr(q[i] - p * q[n + m + i] == 0, "Integer Constraint")
    for i in range(m):
        model.addQConstr(q[n + i] - p * q[n * 2 + m + i] == 0, "Integer Constraint")

    # Solve the Constructed Model
    model.optimize()
    for var in model.getVars():
        if model.status == grb.GRB.OPTIMAL:
            print(var.varName, "\t", var.x)

    total_capacity_weight = 0
    total_capacity_volume = 0
    truck_loading_weight = initial_order_weight
    truck_loading_volume = initial_order_volume
    filler_qty = []
    truck_qty = []

    for var in model.getVars():
        if model.status == grb.GRB.OPTIMAL:
            if var.index == 0:
                p_value = var.x
            if var.index in range(1, n + 1):
                filler_qty.append(var.x / p_value)
                truck_loading_weight += var.x / p_value * order_unit_weight[var.index - 1]
                truck_loading_volume += var.x / p_value * order_unit_volume[var.index - 1]
            elif var.index in range(n + 1, n + m + 1):
                truck_qty.append(var.x / p_value)
                total_capacity_weight += var.x / p_value * truck_capacity_weight[var.index - n - 1]
                total_capacity_volume += var.x / p_value * truck_capacity_volume[var.index - n - 1]
    # model.setParam(grb.GRB.Param.ObjNumber, 0)
    unit_cost = model.objNVal
    cost = unit_cost * max(truck_loading_weight, truck_loading_volume / 3)
    # cost = sum(np.multiply(truck_cost, truck_qty))
    pt = max(truck_loading_weight, truck_loading_volume / 3)
    # unit_cost = cost / pt
    wfr = truck_loading_weight / total_capacity_weight
    vfr = truck_loading_volume / total_capacity_volume
    mix = truck_loading_volume / truck_loading_weight

    return filler_qty, truck_qty, unit_cost, cost, pt, wfr, vfr, mix


def order_shaping(ao_order_data, order_data, truck_data):
    # Data Initialize
    order_data = pd.concat(
        [order_data.groupby(by=["shipto", "material_num"])[["CS", "max_filler_CS"]].sum(),
         order_data.groupby(by=["shipto", "material_num"])[["category", "volume_cube", "weight_ton", "仓报价"]].max()], axis=1) # 该步骤会去掉material_num为空的行

    order_data["category"] = order_data["category"].astype(str)
    order_data = order_data.rename(columns={"volume_cube": "unit_volume_cube", "weight_ton": "unit_weight_ton", "仓报价": "unit_price"})
    order_data["unit_volume_cube"] = order_data["unit_volume_cube"].astype(float)
    order_data["unit_weight_ton"] = order_data["unit_weight_ton"].astype(float)
    order_data["unit_price"] = order_data["unit_price"].astype(float)
    order_data["volume_cube"] = order_data["CS"] * order_data["unit_volume_cube"]
    order_data["weight_ton"] = order_data["CS"] * order_data["unit_weight_ton"]
    order_data["price"] = order_data["CS"] * order_data["unit_price"]
    order_data = order_data.reset_index(names=["shipto", "material_num"])
    order_data["shipto"] = order_data["shipto"].astype(str)
    order_data["material_num"] = order_data["material_num"].astype(str)
    if len(ao_order_data) > 0:
        ao_order_data = pd.concat(
            [ao_order_data.groupby(by=["shipto", "material_num"])[["CS"]].sum(),
             ao_order_data.groupby(by=["shipto", "material_num"])[["category", "volume_cube", "weight_ton", "仓报价"]].max()], axis=1)
        ao_order_data["category"] = ao_order_data["category"].astype(str)
        ao_order_data = ao_order_data.rename(columns={"volume_cube": "unit_volume_cube", "weight_ton": "unit_weight_ton", "仓报价": "unit_price"})
        ao_order_data["unit_volume_cube"] = ao_order_data["unit_volume_cube"].astype(float)
        ao_order_data["unit_weight_ton"] = ao_order_data["unit_weight_ton"].astype(float)
        ao_order_data["unit_price"] = ao_order_data["unit_price"].astype(float)
        ao_order_data["volume_cube"] = ao_order_data["CS"] * ao_order_data["unit_volume_cube"]
        ao_order_data["weight_ton"] = ao_order_data["CS"] * ao_order_data["unit_weight_ton"]
        ao_order_data["price"] = ao_order_data["CS"] * ao_order_data["unit_price"]
        ao_order_data = ao_order_data.reset_index(names=["shipto", "material_num"])
        ao_order_data["shipto"] = ao_order_data["shipto"].astype(str)
        ao_order_data["material_num"] = ao_order_data["material_num"].astype(str)

    # Parameters
    leave_order_data = pd.DataFrame()
    if len(order_data[(order_data["volume_cube"].isnull()) | (order_data["weight_ton"].isnull()) | (order_data["price"].isnull())]) > 0:
        leave_order_data = pd.concat([leave_order_data, order_data[(order_data["weight_ton"].isnull()) | (order_data["volume_cube"].isnull()) | (order_data["price"].isnull())]])
        order_data = order_data[(order_data["weight_ton"].notnull()) & (order_data["volume_cube"].notnull()) & (order_data["price"].notnull())]

    if len(ao_order_data) > 0:
        initial_order_weight = ao_order_data["weight_ton"].sum() + order_data["weight_ton"].sum()
        initial_order_volume = ao_order_data["volume_cube"].sum() + order_data["volume_cube"].sum()
    else:
        initial_order_weight = order_data["weight_ton"].sum()
        initial_order_volume = order_data["volume_cube"].sum()

    if len(order_data["material_num"]) >= 80:
        material = list(order_data["material_num"])
        original_order_data = order_data
        order_data = order_data.nlargest(80, "max_filler_CS")
        selected_material = list(order_data["material_num"])
        not_selected_material = list(set(material) - set(selected_material))
        leave_order_data = pd.concat([leave_order_data, original_order_data[original_order_data["material_num"].isin(not_selected_material)]])

    order_data.sort_values(by=["category"], inplace=True)
    order_data = order_data.reset_index()
    order_unit_weight = np.array(order_data["unit_weight_ton"])
    order_unit_volume = np.array(order_data["unit_volume_cube"])
    order_unit_price = np.array(order_data["unit_price"])
    max_qty = np.array(order_data["max_filler_CS"])
    min_qty = - max_qty
    category_list = order_data["category"]
    material_list = order_data["material_num"]
    category_key = list(order_data["category"].drop_duplicates())
    category_index_lb = []
    category_index_ub = []
    category_price = []
    lb = 0
    ub = 0
    for i in range(len(category_list) - 1):
        if category_list[i] != category_list[i + 1]:
            ub = i
            category_index_lb.append(lb)
            category_index_ub.append(ub)
            category_price.append(order_data.loc[lb:ub, "price"].sum())
            lb = i + 1
        if i == len(category_list) - 2:
            ub = i + 1
            category_index_lb.append(lb)
            category_index_ub.append(ub)
            category_price.append(order_data.loc[lb:ub, "price"].sum())

    if len(ao_order_data) > 0:
        for ctgry in category_key:
            if len(ao_order_data[ao_order_data["category"] == ctgry]) > 0:
                category_price[category_key.index(ctgry)] += ao_order_data[ao_order_data["category"] == ctgry]["price"].sum()

    if len(leave_order_data) > 0:
        leave_category_list = leave_order_data["category"]
        leave_material_list = leave_order_data["material_num"]
        leave_max_qty = np.array(leave_order_data["max_filler_CS"])
        leave_filler_qty = np.zeros(len(leave_material_list))
    else:
        leave_category_list = []
        leave_material_list = []
        leave_max_qty = []
        leave_filler_qty = []

    if "Priority" in list(order_data.columns):
        priority_param = np.array(order_data["Priority"])
    else:
        priority_param = np.zeros(len(order_data["material_num"]))

    truck_capacity_weight = np.array(truck_data["Weight Capacity"])
    truck_capacity_volume = np.array(truck_data["Max Load Volume"])
    truck_cost = np.array(truck_data["Base Charge"])

    available_truck_type = np.array(truck_data[truck_data["Available Truck Type"] == "Y"]["Truck Type"])
    available_truck_capacity_weight = np.array(truck_data[truck_data["Available Truck Type"] == "Y"]["Weight Capacity"])
    available_truck_capacity_volume = np.array(truck_data[truck_data["Available Truck Type"] == "Y"]["Max Load Volume"])
    available_truck_cost = np.array(truck_data[truck_data["Available Truck Type"] == "Y"]["Base Charge"])

    # Calculate
    base_truck_qty, base_cost, base_unit_cost, base_pt, base_wfr, base_vfr, base_mix = baseline(initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume, truck_cost)
    filler_qty, truck_qty, unit_cost, cost, pt, wfr, vfr, mix = model_execution(initial_order_weight, initial_order_volume,
                                                                                      order_unit_weight, order_unit_volume,
                                                                                      order_unit_price, category_key,
                                                                                      category_index_lb, category_index_ub,
                                                                                      category_price, max_qty, min_qty,
                                                                                      priority_param, available_truck_capacity_weight,
                                                                                      available_truck_capacity_volume, available_truck_cost)

    if len(leave_order_data) > 0:
        category_list = pd.concat([category_list, leave_category_list])
        material_list = pd.concat([material_list, leave_material_list])
        max_qty = np.concatenate((max_qty, leave_max_qty))
        max_qty = np.array([int(qty) for qty in max_qty])
        filler_qty = np.concatenate((filler_qty, leave_filler_qty))

    return base_truck_qty, base_cost, base_unit_cost, base_pt, base_wfr, base_vfr, base_mix, category_list, material_list, max_qty, filler_qty, truck_qty, unit_cost, cost, pt, wfr, vfr, mix


# Calculation Execution and Display Result
if exist_order_flag == "Y":
    order_data_result = pd.DataFrame()
    shipto_list = ["Ship-to: " + shipto for shipto in list(selected_order_data["shipto"].unique())]
    shipto_num = len(shipto_list)
    if shipto_num > 0:
        tab_vars = ["tab_" + str(i) for i in range(shipto_num)]
        tab_vars = st.tabs(shipto_list)
        filler_rate_vars = ["filler_rate_" + str(i) for i in range(shipto_num)]
        best_filler_rate_vars = ["best_filler_rate_" + str(i) for i in range(shipto_num)]
    else:
        st.warning("**Warning:**" + " No matched Ship-to, the optimization terminated.")
        st.stop()
    label = 0
    for shipto in np.array(selected_order_data["shipto"].unique()):
        order_data = selected_order_data[selected_order_data["shipto"] == shipto]
        by_category_result = order_data.groupby(by=["category"])[["CS"]].sum()
        by_category_result = by_category_result.rename(columns={"CS": "Before Shaping CS (#)"})
        if len(original_ao_order_data) > 0:
            ao_order_data = selected_ao_order_data[selected_ao_order_data["shipto"] == shipto]
            if len(ao_order_data) > 0:
                initial_order_qty = ao_order_data["CS"].sum() + order_data["CS"].sum()
            else:
                initial_order_qty = order_data["CS"].sum()
        else:
            initial_order_qty = order_data["CS"].sum()
            ao_order_data = pd.DataFrame()
        truck_data = truck_master[truck_master["Ship-to"] == shipto]
        ideal_truck_type_data = ideal_truck_type_master[ideal_truck_type_master["Ship-to"] == shipto]
        index = ideal_truck_type_data.index.tolist()[0]
        ideal_truck_type = ideal_truck_type_data["Ideal Truck Type"][index]
        ideal_unit_cost = ideal_truck_type_data["Ideal Unit Cost"][index]
        ideal_vfr = ideal_truck_type_data["Ideal VFR"][index]
        ideal_wfr = ideal_truck_type_data["Ideal WFR"][index]
        ideal_mix = ideal_truck_type_data["Ideal Mix"][index]
        customer_name = ideal_truck_type_data["Customer Name"][index]

        truck_type = np.array(truck_data["Truck Type"])
        available_truck_type = np.array(truck_data[truck_data["Available Truck Type"] == "Y"]["Truck Type"])

        base_truck_qty, base_cost, base_unit_cost, base_pt, base_wfr, base_vfr, base_mix, category_list, material_list, max_qty, filler_qty, truck_qty, unit_cost, cost, pt, wfr, vfr, mix = order_shaping(ao_order_data, order_data, truck_data)

        min_qty = - max_qty
        order_data_result_split = pd.DataFrame(
            {
                "material_num": material_list,
                "filler_qty": filler_qty
            }
        )
        order_data_result_split["shipto"] = shipto
        order_data_result = pd.concat([order_data_result, order_data_result_split])
        material_total = len(filler_qty)
        material_changed = len([qty for qty in filler_qty if qty is not None and qty != "" and qty != 0])
        material_changed_percent = material_changed / material_total

        base_loss = (ideal_unit_cost - base_unit_cost) * base_pt
        loss = (ideal_unit_cost - unit_cost) * pt
        if base_loss > 0:
            base_loss = 0
        if loss > 0:
            loss = 0
        base_loss_percent = base_loss / base_cost
        loss_percent = loss / cost

        size_of_prize = loss - base_loss
        size_of_prize_percent = (loss - base_loss) / base_cost
        base_truck_selected = ""
        for i in range(len(base_truck_qty)):
            if base_truck_qty[i] > 0:
                if base_truck_selected == "":
                    base_truck_selected += str(truck_type[i]) + "*" + "{:.0f}".format(base_truck_qty[i])
                else:
                    base_truck_selected += "+" + str(truck_type[i]) + "*" + "{:.0f}".format(base_truck_qty[i])
        base_truck_num = sum(base_truck_qty)
        if initial_order_qty >= 3500:
            base_slog = "2.3%"
        elif initial_order_qty >= 2000:
            base_slog = "2.1%"
        elif initial_order_qty >= 800:
            base_slog = "1.5%"
        else:
            base_slog = "0%"

        saving = (base_unit_cost - unit_cost) * pt
        saving_percent = (base_unit_cost - unit_cost) / base_unit_cost
        filler_qty = np.array([round(qty) for qty in filler_qty])
        qty_changed = np.sum(filler_qty)
        abs_qty_changed = np.sum(np.abs(filler_qty))
        order_qty = initial_order_qty + qty_changed
        qty_changed_percent = qty_changed / initial_order_qty
        abs_qty_changed_percent = abs_qty_changed / initial_order_qty
        by_category_filler_qty = pd.DataFrame({"category": category_list, "After Shaping CS (#)": filler_qty})
        by_category_filler_qty = by_category_filler_qty.groupby(by=["category"])[["After Shaping CS (#)"]].sum()
        by_category_result = pd.merge(by_category_result, by_category_filler_qty, how="left", on="category")
        by_category_result["After Shaping CS (#)"] = by_category_result["Before Shaping CS (#)"] + by_category_result["After Shaping CS (#)"]
        if adopt_calculation_flag != "Y":
            by_category_result = by_category_result.reset_index().rename(columns={"category": "Category"})

        truck_selected = ""
        for i in range(len(truck_qty)):
            if truck_qty[i] > 0:
                if truck_selected == "":
                    truck_selected += str(available_truck_type[i]) + "*" + "{:.0f}".format(truck_qty[i])
                else:
                    truck_selected += "+" + str(available_truck_type[i]) + "*" + "{:.0f}".format(truck_qty[i])
        truck_num = sum(truck_qty)
        if order_qty >= 3500:
            slog = "2.3%"
        elif order_qty >= 2000:
            slog = "2.1%"
        elif order_qty >= 800:
            slog = "1.5%"
        else:
            slog = "0%"

        if adopt_calculation_flag == "Y":
            by_category_adopt_filler_qty = pd.DataFrame(
                {"category": order_data["category"], "Adopted Shaping CS (#)": order_data["adopt_qty"] / order_data["箱规"]})
            by_category_adopt_filler_qty = by_category_adopt_filler_qty.groupby(by=["category"])[["Adopted Shaping CS (#)"]].sum()
            by_category_result = pd.merge(by_category_result, by_category_adopt_filler_qty, how="left", on="category")
            by_category_result = by_category_result.reset_index().rename(columns={"category": "Category", "After Shaping CS (#)": "Proposed Shaping CS (#)"})

            qty = order_data["adopt_qty"] / order_data["箱规"]
            weight = order_data["weight_ton"] * qty
            volume = order_data["volume_cube"] * qty
            if len(ao_order_data) > 0:
                ao_qty = ao_order_data["CS"]
                ao_weight = ao_order_data["weight_ton"] * ao_qty
                ao_volume = ao_order_data["volume_cube"] * ao_qty
                initial_order_weight = ao_weight.sum() + weight.sum()
                initial_order_volume = ao_volume.sum() + volume.sum()
                adopt_order_qty = ao_qty.sum() + qty.sum()
            else:
                initial_order_weight = weight.sum()
                initial_order_volume = volume.sum()
                adopt_order_qty = qty.sum()

            adopt_filler_qty = order_data["adopt_qty"] / order_data["箱规"] - order_data["CS"]
            adopt_qty_changed = np.sum(adopt_filler_qty)
            adopt_abs_qty_changed = np.sum(np.abs(adopt_filler_qty))
            adopt_qty_changed_percent = adopt_qty_changed / initial_order_qty
            adopt_abs_qty_changed_percent = adopt_abs_qty_changed / initial_order_qty
            adopt_material_changed = len(
                [qty for qty in adopt_filler_qty if qty is not None and qty != "" and qty != 0])
            adopt_material_changed_percent = adopt_material_changed / material_total

            truck_capacity_weight = np.array(truck_data["Weight Capacity"])
            truck_capacity_volume = np.array(truck_data["Max Load Volume"])
            truck_cost = np.array(truck_data["Base Charge"])

            adopt_truck_qty, adopt_cost, adopt_unit_cost, adopt_pt, adopt_wfr, adopt_vfr, adopt_mix = baseline(
                initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume,
                truck_cost)

            adopt_loss = (ideal_unit_cost - adopt_unit_cost) * adopt_pt
            if adopt_loss > 0:
                adopt_loss = 0
            adopt_loss_percent = adopt_loss / adopt_cost

            adopt_saving = (base_unit_cost - adopt_unit_cost) * adopt_pt
            adopt_saving_percent = (base_unit_cost - adopt_unit_cost) / base_unit_cost

            adopt_truck_selected = ""
            for i in range(len(adopt_truck_qty)):
                if adopt_truck_qty[i] > 0:
                    if adopt_truck_selected == "":
                        adopt_truck_selected += str(truck_type[i]) + "*" + "{:.0f}".format(adopt_truck_qty[i])
                    else:
                        adopt_truck_selected += "+" + str(truck_type[i]) + "*" + "{:.0f}".format(adopt_truck_qty[i])
            adopt_truck_num = sum(adopt_truck_qty)
            if adopt_order_qty >= 3500:
                adopt_slog = "2.3%"
            elif adopt_order_qty >= 2000:
                adopt_slog = "2.1%"
            elif adopt_order_qty >= 800:
                adopt_slog = "1.5%"
            else:
                adopt_slog = "0%"


        with tab_vars[label]:
            st.info("Basic Info")
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Ship-to")
                    st.markdown("Customer")
                with col2:
                    st.markdown(shipto)
                    st.markdown(customer_name)

            st.info("Order Shaping Top Line Result")
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Result**")
                    st.markdown("Size of Prize (RMB)")
                    st.markdown("Quantity Changed (CS)")
                    st.markdown("Abs Quantity Changed (CS)")
                with col2:
                    st.markdown("**Absolute**")
                    st.markdown("{:.0f}".format(size_of_prize))
                    st.markdown(qty_changed)
                    st.markdown(abs_qty_changed)
                with col3:
                    st.markdown("**%**")
                    st.markdown("{:.0f}%".format(size_of_prize_percent * 100))
                    st.markdown("{:.0f}%".format(qty_changed_percent * 100))
                    st.markdown("{:.0f}%".format(abs_qty_changed_percent * 100))

            result = pd.DataFrame(
                {
                    "Index": ["Unit Cost (RMB/PT)", "Spending (RMB)", "Loss (RMB)", "Loss (%)", "Saving (RMB)",
                              "Saving (%)",
                              "Truck Type", "Truck (#)", "VFR", "WFR", "Heavy/Light Mix(CBM/Ton)", "Quantity (CS)",
                              "Quantity Changed (CS)", "Quantity Changed (%)", "Abs Quantity Changed (CS)",
                              "Abs Quantity Changed (%)",
                              "SLOG Impact", "SKU (#)", "SKU Changed (#)", "SKU Changed (%)"],
                    "Before Shaping": ["{:.1f}".format(base_unit_cost), "{:.0f}".format(base_cost),
                                       "{:.0f}".format(base_loss), "{:.0f}%".format(base_loss_percent * 100), "N/A",
                                       "N/A", base_truck_selected, "{:.0f}".format(base_truck_num),
                                       "{:.0f}%".format(base_vfr * 100), "{:.0f}%".format(base_wfr * 100),
                                       "{:.1f}".format(base_mix), initial_order_qty, "N/A", "N/A", "N/A", "N/A",
                                       base_slog,
                                       material_total, "N/A", "N/A"],
                    "After Shaping": ["{:.1f}".format(unit_cost), "{:.0f}".format(cost), "{:.0f}".format(loss),
                                      "{:.0f}%".format(loss_percent * 100),
                                      "{:.0f}".format(saving), "{:.0f}%".format(saving_percent * 100),
                                      truck_selected, "{:.0f}".format(truck_num), "{:.0f}%".format(vfr * 100), "{:.0f}%".format(wfr * 100),
                                      "{:.1f}".format(mix), order_qty, qty_changed,
                                      "{:.0f}%".format(qty_changed_percent * 100), abs_qty_changed,
                                      "{:.0f}%".format(abs_qty_changed_percent * 100), slog,
                                      material_total, material_changed,
                                      "{:.0f}%".format(material_changed_percent * 100)],
                    "Ideal State": ["{:.1f}".format(ideal_unit_cost), "N/A", "0", "0%", "N/A", "N/A", ideal_truck_type, "N/A",
                                    "{:.0f}%".format(ideal_vfr * 100), "{:.0f}%".format(ideal_wfr * 100),
                                    "{:.1f}".format(ideal_mix), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                                    "N/A"]
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
                ],
                "enableRangeSelection": True
            }

            if adopt_calculation_flag == "Y":
                result = pd.DataFrame(
                    {
                        "Index": ["Unit Cost (RMB/PT)", "Payable Ton (PT)", "Spending (RMB)", "Loss (RMB)", "Loss (%)", "Saving (RMB)",
                                  "Saving (%)",
                                  "Truck Type", "Truck (#)", "VFR", "WFR", "Heavy/Light Mix(CBM/Ton)", "Quantity (CS)",
                                  "Quantity Changed (CS)", "Quantity Changed (%)", "Abs Quantity Changed (CS)",
                                  "Abs Quantity Changed (%)",
                                  "SLOG Impact", "SKU (#)", "SKU Changed (#)", "SKU Changed (%)"],
                        "Before Shaping": ["{:.1f}".format(base_unit_cost), "{:.1f}".format(base_pt), "{:.0f}".format(base_cost),
                                           "{:.0f}".format(base_loss), "{:.0f}%".format(base_loss_percent * 100), "N/A",
                                           "N/A", base_truck_selected, "{:.0f}".format(base_truck_num),
                                           "{:.0f}%".format(base_vfr * 100), "{:.0f}%".format(base_wfr * 100),
                                           "{:.1f}".format(base_mix), initial_order_qty, "N/A", "N/A", "N/A", "N/A",
                                           base_slog,
                                           material_total, "N/A", "N/A"],
                        "Proposed Shaping": ["{:.1f}".format(unit_cost), "{:.1f}".format(pt), "{:.0f}".format(cost), "{:.0f}".format(loss),
                                             "{:.0f}%".format(loss_percent * 100),
                                             "{:.0f}".format(saving), "{:.0f}%".format(saving_percent * 100),
                                             truck_selected, "{:.0f}".format(truck_num), "{:.0f}%".format(vfr * 100), "{:.0f}%".format(wfr * 100),
                                             "{:.1f}".format(mix), order_qty, qty_changed,
                                             "{:.0f}%".format(qty_changed_percent * 100), abs_qty_changed,
                                             "{:.0f}%".format(abs_qty_changed_percent * 100), slog,
                                             material_total, material_changed,
                                             "{:.0f}%".format(material_changed_percent * 100)],
                        "Adopted Shaping": ["{:.1f}".format(adopt_unit_cost), "{:.1f}".format(adopt_pt), "{:.0f}".format(adopt_cost),
                                            "{:.0f}".format(adopt_loss), "{:.0f}%".format(adopt_loss_percent * 100),
                                            "{:.0f}".format(adopt_saving), "{:.0f}%".format(adopt_saving_percent * 100),
                                            adopt_truck_selected, "{:.0f}".format(adopt_truck_num),
                                            "{:.0f}%".format(adopt_vfr * 100), "{:.0f}%".format(adopt_wfr * 100),
                                            "{:.1f}".format(adopt_mix), adopt_order_qty, adopt_qty_changed,
                                            "{:.0f}%".format(adopt_qty_changed_percent * 100), adopt_abs_qty_changed,
                                            "{:.0f}%".format(adopt_abs_qty_changed_percent * 100), adopt_slog,
                                            material_total, adopt_material_changed,
                                            "{:.0f}%".format(adopt_material_changed_percent * 100)],
                        "Ideal State": ["{:.1f}".format(ideal_unit_cost), "N/A", "N/A", "0", "0%", "N/A", "N/A",
                                        ideal_truck_type, "N/A",
                                        "{:.0f}%".format(ideal_vfr * 100), "{:.0f}%".format(ideal_wfr * 100),
                                        "{:.1f}".format(ideal_mix), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                                        "N/A", "N/A"]
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
                            "headerName": "Proposed Shaping",
                            "field": "Proposed Shaping",
                            "cellStyle": {
                                "text-align": "center"
                            },
                            "sortable": False
                        },
                        {
                            "headerName": "Adopted Shaping",
                            "field": "Adopted Shaping",
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
                    ],
                    "enableRangeSelection": True
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
                    "Category": category_list,
                    "SKU Number": material_list,
                    "SKU Quantity Limit (-MIN CS, +MAX CS)": ["(" + "{:.1f}".format(min_qty) + ", +" + "{:.1f}".format(max_qty) + ")" for min_qty, max_qty in zip(min_qty, max_qty)],
                    "Reco Quantity (CS)": np.where(filler_qty <= 0, filler_qty, ["+" + str(qty) for qty in filler_qty])
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
                ],
                "enableRangeSelection": True
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

            by_category_result_grid_options = {
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
                        "headerName": "Before Shaping CS (#)",
                        "field": "Before Shaping CS (#)",
                        "filter": "agTextColumnFilter",
                        # "suppressSizeToFit": True,
                        "cellStyle": {
                            "text-align": "center"
                        }
                    },
                    {
                        "headerName": "After Shaping CS (#)",
                        "field": "After Shaping CS (#)",
                        "filter": "agTextColumnFilter",
                        # "suppressSizeToFit": True,
                        "cellStyle": {
                            "text-align": "center"
                        }
                    }
                ],
                "enableRangeSelection": True
            }

            if adopt_calculation_flag == "Y":
                by_category_result_grid_options = {
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
                            "headerName": "Before Shaping CS (#)",
                            "field": "Before Shaping CS (#)",
                            "filter": "agTextColumnFilter",
                            # "suppressSizeToFit": True,
                            "cellStyle": {
                                "text-align": "center"
                            }
                        },
                        {
                            "headerName": "Proposed Shaping CS (#)",
                            "field": "Proposed Shaping CS (#)",
                            "filter": "agTextColumnFilter",
                            # "suppressSizeToFit": True,
                            "cellStyle": {
                                "text-align": "center"
                            }
                        },
                        {
                            "headerName": "Adopted Shaping CS (#)",
                            "field": "Adopted Shaping CS (#)",
                            "filter": "agTextColumnFilter",
                            # "suppressSizeToFit": True,
                            "cellStyle": {
                                "text-align": "center"
                            }
                        }
                    ],
                    "enableRangeSelection": True
                }

            st.info("By Category Result")
            AgGrid(
                by_category_result,
                gridOptions=by_category_result_grid_options,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                fit_columns_on_grid_load=True,
                theme="alpine",
                custom_css=custom_css
            )
        label += 1

    st.divider()
    st.info("Recommendation Data Download")
    selected_order_data["suggest_qty"] = 0
    not_selected_order_data["suggest_qty"] = np.nan

    if order_data_result.empty:
        st.warning("**Warning:**" + "There is no order data within the testing scope in the file(s).")
    else:
        for shipto in np.array(selected_order_data["shipto"].unique()):
            for material_num in order_data_result["material_num"]:
                order_data_related = selected_order_data[(selected_order_data["material_num"] == material_num) & (selected_order_data["shipto"] == shipto)]
                if len(order_data_related) == 1:
                    selected_order_data.loc[(selected_order_data["material_num"] == material_num) & (selected_order_data["shipto"] == shipto), "suggest_qty"] = list(order_data_result[(order_data_result["material_num"] == material_num) & (order_data_result["shipto"] == shipto)]["filler_qty"])
                else:
                    qty_list = list(order_data_related["采购需求数量*"])
                    qty_sum = order_data_related["采购需求数量*"].sum()
                    total_qty = order_data_result[(order_data_result["material_num"] == material_num) & (order_data_result["shipto"] == shipto)]["filler_qty"]
                    final_qty = []
                    current_sum = 0
                    for i in range(len(order_data_related)):
                        if i < len(order_data_related) - 1:
                            final_qty.append(round(total_qty / qty_sum * qty_list[i]))
                            current_sum += round(total_qty / qty_sum * qty_list[i])
                        else:
                            final_qty.append(total_qty - current_sum)
                    selected_order_data.loc[(selected_order_data["material_num"] == material_num) & (selected_order_data["shipto"] == shipto), "suggest_qty"] = final_qty

        selected_order_data["suggest_qty_in_cs"] = np.round(selected_order_data["suggest_qty"])
        selected_order_data["suggest_qty"] = np.round(selected_order_data["suggest_qty"] * selected_order_data["箱规"])

    for source in upload_source_list:
        if source in download_source_list:
            source_selected_order_data = selected_order_data[selected_order_data["Source"] == source]
            selected_city = list(source_selected_order_data["Region"].drop_duplicates())
            source_not_selected_order_data = not_selected_order_data[not_selected_order_data["Source"] == source]
            output_data = pd.concat([source_selected_order_data, source_not_selected_order_data])
            output_data.drop(["补货前周转天数", "补货后周转天数", "正负可调整周转天数"], axis=1, inplace=True)

            source = source.rstrip("_竖版")
            index = output_data.columns.tolist().index("采购需求数量*")
            if "建议调整数量" in output_data.columns.tolist():
                output_data.drop(["建议调整数量"], axis=1, inplace=True)
                output_data.drop(["建议调整箱数"], axis=1, inplace=True)
            output_data.insert(index + 1, "建议调整数量", output_data.pop("suggest_qty"))
            output_data.insert(index + 2, "建议调整箱数", output_data.pop("suggest_qty_in_cs"))
            if "实际采纳数量" in output_data.columns.tolist():
                output_data = output_data.rename(columns={"实际采纳数量": "实际采纳数量_temp"})
                output_data.insert(index + 3, "实际采纳数量", output_data.pop("实际采纳数量_temp"))
            else:
                output_data.insert(index + 3, "实际采纳数量", np.nan)
            city_col_name = [column for column in column_dict[source] if "配送中心" in column][0]
            output_data = output_data.rename(columns={"京东码": "sku*", "配送中心*(格式：北京,上海,广州)": city_col_name})
            column_list = output_data.columns.tolist()
            keep_column_list = column_dict[source] + ["建议调整数量", "建议调整箱数", "实际采纳数量"]
            drop_column_list = [column for column in column_list if column not in keep_column_list]
            output_data.drop(drop_column_list, axis=1, inplace=True)

            output = BytesIO()
            if source.endswith(".xlsx"):
                file_name = source.rstrip(".xlsx") + "_result.xlsx"
            elif source.endswith(".xls"):
                file_name = source.rstrip(".xls") + "_result.xlsx"
            else:
                file_name = "unknown_file_type_result.xlsx"
            with pd.ExcelWriter(output) as writer:
                output_data.to_excel(writer, index=False)
            st.download_button(
                label="Download Optimization Result for " + source,
                data=output.getvalue(),
                file_name=file_name,
                mime="application/vnd.ms-excel"
            )
        else:
            if "横版" in source:
                source = source.rstrip("_横版")
            else:
                source = source.rstrip("_竖版")
            st.warning("**Warning:**" + "There is no order data within the testing scope in the file " + source)
