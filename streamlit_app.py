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
st.title("Order Shaping Tool")
st.caption("Feel free to contact developer _Wang Nan_ if you have any question: wang.n.22@pg.com")
st.divider()


# Read Master Data
sharepointUsername = "wang.n.22@pg.com"
sharepointPassword = "POIUytrEWQ#2339"
sharepointSite = "https://pgone.sharepoint.com/sites/GCInnovationandCapabilityTeam"
website = "https://pgone.sharepoint.com"

authcookie = Office365(website, username=sharepointUsername, password=sharepointPassword).GetCookies()
site = Site(sharepointSite, version=Version.v365, authcookie=authcookie)
folder = site.Folder("Shared Documents/31. Order Loss Analysis/JD Full Truck Load/Order Shaping Tool/Input_MD")
truck_master_filename = "Tariff.xlsx"
ideal_truck_type_master_filename = "Ideal Truck Type.xlsx"
sku_transfer_data_filename = "京东直供数据2024_0912.xlsx"
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

sku_transfer_data = pd.read_excel("sku_transfer_data_temp.xlsx", dtype={"京东码": str, "宝洁码": str})

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
# selected_shipto = ["2003213268"]
selected_shipto = ["2003213268", "2002921387"]
exist_order_flag = "N"


# Rate Definition
filler_rate = st.slider(label="Please select the filler rate (%):", min_value=0, max_value=100, value=10, step=1)
filler_rate = filler_rate / 100


# Data Initialization
if len(uploaded_files) > 0:
    for uploaded_file in uploaded_files:
        data_split = pd.read_excel(uploaded_file)
        if True in data_split.columns.str.contains("POA回告数量"):
            original_ao_order_data_split = data_split.astype({"Ship to编码": str, "宝洁八位码": str})
            original_ao_order_data_split = original_ao_order_data_split.rename(columns={"Ship to编码": "shipto", "宝洁八位码": "material_num", "POA回告数量（箱数）": "CS"})
            original_ao_order_data_split = pd.merge(original_ao_order_data_split, sku_master.loc[:, ["material_num", "category", "volume_cube", "weight_ton"]], how="left", on="material_num")
            original_ao_order_data = pd.concat([original_ao_order_data, original_ao_order_data_split])
        else:
            exist_order_flag = "Y"
            if True in data_split.columns.str.contains("配送中心"):
                city_col_name = list(data_split.columns[data_split.columns.str.contains("配送中心")])[0]
                original_order_data_split = data_split.astype({"sku*": str, city_col_name: str})
                order_data_split = original_order_data_split.rename(columns={"sku*": "京东码", city_col_name: "配送中心*(格式：北京,上海,广州)"})
                order_data_split["Source"] = uploaded_file.name + "_竖版"
            else:
                original_order_data_split = data_split.astype({"商品编码": str})
                # order_data_split = pd.melt(original_order_data_split, id_vars="商品编码", value_vars=["北京"], var_name="配送中心*(格式：北京,上海,广州)", value_name="采购需求数量*")
                order_data_split = original_order_data_split.rename(columns={"商品编码": "京东码", "北京": "采购需求数量*"})
                order_data_split["配送中心*(格式：北京,上海,广州)"] = "北京"
                order_data_split["Source"] = uploaded_file.name + "_横版"
            order_data_split = pd.merge(order_data_split, sku_transfer_data.loc[:, ["京东码", "宝洁码", "箱规⑥"]], how="left", on="京东码")
            order_data_split["CS"] = order_data_split["采购需求数量*"] / order_data_split["箱规⑥"]
            order_data_split["max_filler_CS"] = order_data_split["采购需求数量*"] / order_data_split["箱规⑥"] * filler_rate
            order_data_split = order_data_split.rename(columns={"宝洁码": "material_num"})
            order_data_split = pd.merge(order_data_split, sku_master.loc[:, ["material_num", "category", "volume_cube", "weight_ton"]], how="left", on="material_num")
            order_data_split["Region"] = order_data_split["配送中心*(格式：北京,上海,广州)"]
            category = np.array(order_data_split[order_data_split["category"].notnull()]["category"])
            source_shipto_city_data = shipto_city_data[shipto_city_data["品类"].str.startswith(category[0] + "/", na=False) | shipto_city_data["品类"].str.contains("/" + category[0], na=False)]
            order_data_split = pd.merge(order_data_split, source_shipto_city_data.loc[:, ["Region", "shipto"]], how="left", on="Region")
            order_data = pd.concat([order_data, order_data_split])
    if exist_order_flag == "Y":
        upload_source_list = list(set(list(order_data["Source"])))
        not_selected_shipto = list(set(list(order_data["shipto"])) - set(selected_shipto))
        not_selected_order_data = order_data[(order_data["shipto"].isin(not_selected_shipto)) & (order_data["采购需求数量*"] == 0)]
        selected_order_data = order_data[(order_data["shipto"].isin(selected_shipto)) & (order_data["采购需求数量*"] != 0)]
        download_source_list = list(set(list(selected_order_data["Source"])))
        missing_sku_list = ", ".join("{0}".format(sku) for sku in list(selected_order_data[(selected_order_data["volume_cube"].isnull()) | (selected_order_data["weight_ton"].isnull())]["京东码"]))
        if len(missing_sku_list) > 0:
            st.warning("**Warning:**" + " Missing weight/volume master data of SKU " + missing_sku_list + " for the order data, which will be excluded from the optimization.")
        if len(original_ao_order_data) > 0:
            not_selected_ao_order_data = original_ao_order_data[(original_ao_order_data["shipto"].isin(not_selected_shipto)) & (original_ao_order_data["CS"] == 0)]
            selected_ao_order_data = original_ao_order_data[(original_ao_order_data["shipto"].isin(selected_shipto)) & (original_ao_order_data["CS"] != 0)]
            missing_ao_sku_list = ", ".join("{0}".format(sku) for sku in list(selected_ao_order_data[(selected_ao_order_data["volume_cube"].isnull()) | (selected_ao_order_data["weight_ton"].isnull())]["material_num"]))
            if len(missing_ao_sku_list) > 0:
                st.warning("**Warning:**" + " Missing weight/volume master data of SKU " + missing_ao_sku_list + " for the AO order data, which will be excluded from the optimization.")
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
                    order_unit_volume, max_qty, min_qty, priority_param, truck_capacity_weight, truck_capacity_volume,
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
            q[i] = model.addVar(lb=min_qty[i] / max_pt, ub=max_qty[i] / min_pt, vtype=grb.GRB.CONTINUOUS, name="q_" + str(i))
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
         order_data.groupby(by=["shipto", "material_num"])[["category", "volume_cube", "weight_ton"]].max()], axis=1)
    order_data["category"] = order_data["category"].astype(str)
    order_data = order_data.rename(columns={"volume_cube": "unit_volume_cube", "weight_ton": "unit_weight_ton"})
    order_data["unit_volume_cube"] = order_data["unit_volume_cube"].astype(float)
    order_data["unit_weight_ton"] = order_data["unit_weight_ton"].astype(float)
    order_data["volume_cube"] = order_data["CS"] * order_data["unit_volume_cube"]
    order_data["weight_ton"] = order_data["CS"] * order_data["unit_weight_ton"]
    order_data = order_data.reset_index(names=["shipto", "material_num"])
    order_data["shipto"] = order_data["shipto"].astype(str)
    order_data["material_num"] = order_data["material_num"].astype(str)
    if len(ao_order_data) > 0:
        ao_order_data = pd.concat(
            [ao_order_data.groupby(by=["shipto", "material_num"])[["CS"]].sum(),
             ao_order_data.groupby(by=["shipto", "material_num"])[["category", "volume_cube", "weight_ton"]].max()], axis=1)
        ao_order_data["category"] = ao_order_data["category"].astype(str)
        ao_order_data = ao_order_data.rename(columns={"volume_cube": "unit_volume_cube", "weight_ton": "unit_weight_ton"})
        ao_order_data["unit_volume_cube"] = ao_order_data["unit_volume_cube"].astype(float)
        ao_order_data["unit_weight_ton"] = ao_order_data["unit_weight_ton"].astype(float)
        ao_order_data["volume_cube"] = ao_order_data["CS"] * ao_order_data["unit_volume_cube"]
        ao_order_data["weight_ton"] = ao_order_data["CS"] * ao_order_data["unit_weight_ton"]
        ao_order_data = ao_order_data.reset_index(names=["shipto", "material_num"])
        ao_order_data["shipto"] = ao_order_data["shipto"].astype(str)
        ao_order_data["material_num"] = ao_order_data["material_num"].astype(str)

    # Parameters
    leave_order_data = pd.DataFrame()
    if len(order_data[(order_data["volume_cube"].isnull()) | (order_data["weight_ton"].isnull())]) > 0:
        leave_order_data = pd.concat([leave_order_data, order_data[(order_data["weight_ton"].isnull()) | (order_data["volume_cube"].isnull())]])
        order_data = order_data[(order_data["weight_ton"].notnull()) & (order_data["volume_cube"].notnull())]

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

    order_unit_weight = np.array(order_data["unit_weight_ton"])
    order_unit_volume = np.array(order_data["unit_volume_cube"])
    max_qty = np.array(order_data["max_filler_CS"])
    min_qty = - max_qty
    category_list = order_data["category"]
    material_list = order_data["material_num"]
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

    ideal_truck_type = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Truck Type"])
    ideal_truck_capacity_weight = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Weight Capacity"])
    ideal_truck_capacity_volume = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Max Load Volume"])
    ideal_truck_cost = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Base Charge"])

    # Calculate
    base_truck_qty, base_cost, base_unit_cost, base_pt, base_wfr, base_vfr, base_mix = baseline(initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume, truck_cost)
    filler_qty, truck_qty, unit_cost, cost, pt, wfr, vfr, mix = model_execution(initial_order_weight, initial_order_volume,
                                                                                      order_unit_weight,
                                                                                      order_unit_volume, max_qty, min_qty,
                                                                                      priority_param, ideal_truck_capacity_weight,
                                                                                      ideal_truck_capacity_volume, ideal_truck_cost)

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
    tab_vars = ["tab_" + str(i) for i in range(shipto_num)]
    tab_vars = st.tabs(shipto_list)
    label = 0
    for shipto in np.array(selected_order_data["shipto"].unique()):
        order_data = selected_order_data[selected_order_data["shipto"] == shipto]
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
        ideal_unit_cost = ideal_truck_type_data["Ideal Unit Cost"][index]
        ideal_vfr = ideal_truck_type_data["Ideal VFR"][index]
        ideal_wfr = ideal_truck_type_data["Ideal WFR"][index]
        ideal_mix = ideal_truck_type_data["Ideal Mix"][index]
        customer_name = ideal_truck_type_data["Customer Name"][index]

        truck_type = np.array(truck_data["Truck Type"])
        ideal_truck_type = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Truck Type"])

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

        base_loss = (ideal_unit_cost - base_unit_cost) * base_pt
        loss = (ideal_unit_cost - unit_cost) * pt

        size_of_prize = loss - base_loss
        size_of_prize_percent = (loss - base_loss) / base_cost
        base_truck_selected = ""
        for i in range(len(base_truck_qty)):
            if base_truck_qty[i] > 0:
                if base_truck_selected == "":
                    base_truck_selected += str(truck_type[i]) + "*" + "{:.0f}".format(base_truck_qty[i])
                else:
                    base_truck_selected += "+" + str(truck_type[i]) + "*" + "{:.0f}".format(base_truck_qty[i])
        if initial_order_qty >= 3500:
            base_slog = "2.3%"
        elif initial_order_qty >= 2000:
            base_slog = "2.1%"
        elif initial_order_qty >= 800:
            base_slog = "1.5%"
        else:
            base_slog = "0%"

        saving = (base_unit_cost - unit_cost) * pt
        saving_percent = saving / base_cost
        filler_qty = np.array([round(qty) for qty in filler_qty])
        qty_changed = 0
        for i in range(len(filler_qty)):
            qty_changed += filler_qty[i]
        order_qty = initial_order_qty + qty_changed
        qty_changed_percent = qty_changed / initial_order_qty

        truck_selected = ""
        for i in range(len(truck_qty)):
            if truck_qty[i] > 0:
                if truck_selected == "":
                    truck_selected += str(ideal_truck_type[i]) + "*" + "{:.0f}".format(truck_qty[i])
                else:
                    truck_selected += "+" + str(ideal_truck_type[i]) + "*" + "{:.0f}".format(truck_qty[i])
        if order_qty >= 3500:
            slog = "2.3%"
        elif order_qty >= 2000:
            slog = "2.1%"
        elif order_qty >= 800:
            slog = "1.5%"
        else:
            slog = "0%"

        if "实际采纳数量" in order_data.columns.tolist():
            if order_data["实际采纳数量"] is not np.nan:
                qty = (order_data["采购需求数量*"] + order_data["实际采纳数量"].fillna(0)) / order_data["箱规⑥"]
                weight = order_data["weight_ton"] * qty
                volume = order_data["volume_cube"] * qty
                ao_qty = ao_order_data["CS"]
                ao_weight = ao_order_data["weight_ton"] * ao_qty
                ao_volume = ao_order_data["volume_cube"] * ao_qty
                initial_order_weight = ao_weight.sum() + weight.sum()
                initial_order_volume = ao_volume.sum() + volume.sum()
                adopt_order_qty = ao_qty.sum() + qty.sum()

                truck_capacity_weight = np.array(truck_data["Weight Capacity"])
                truck_capacity_volume = np.array(truck_data["Max Load Volume"])
                truck_cost = np.array(truck_data["Base Charge"])

                adopt_truck_qty, adopt_cost, adopt_unit_cost, adopt_pt, adopt_wfr, adopt_vfr, adopt_mix = baseline(
                    initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume,
                    truck_cost)

                adopt_loss = (ideal_unit_cost - adopt_unit_cost) * adopt_pt

                adopt_truck_selected = ""
                for i in range(len(adopt_truck_qty)):
                    if adopt_truck_qty[i] > 0:
                        if adopt_truck_selected == "":
                            adopt_truck_selected += str(truck_type[i]) + "*" + "{:.0f}".format(adopt_truck_qty[i])
                        else:
                            adopt_truck_selected += "+" + str(truck_type[i]) + "*" + "{:.0f}".format(adopt_truck_qty[i])
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
                with col2:
                    st.markdown("**Absolute**")
                    st.markdown("{:.0f}".format(size_of_prize))
                    st.markdown(qty_changed)
                with col3:
                    st.markdown("**%**")
                    st.markdown("{:.0f}%".format(size_of_prize_percent * 100))
                    st.markdown("{:.0f}%".format(qty_changed_percent * 100))

            result = pd.DataFrame(
                {
                    "Index": ["Unit Cost (RMB/PT)", "Spending (RMB)", "Loss (RMB)", "Saving (RMB)", "Saving (%)", "Truck Type", "VFR", "WFR", "Heavy/Light Mix(CBM/Ton)", "Quantity (CS)", "SLOG Impact"],
                    "Before Shaping": ["{:.1f}".format(base_unit_cost), "{:.0f}".format(base_cost), "{:.0f}".format(base_loss), "N/A", "N/A", base_truck_selected, "{:.0f}%".format(base_vfr * 100), "{:.0f}%".format(base_wfr * 100), "{:.1f}".format(base_mix), initial_order_qty, base_slog],
                    "After Shaping": ["{:.1f}".format(unit_cost), "{:.0f}".format(cost), "{:.0f}".format(loss), "{:.0f}".format(saving), "{:.0f}%".format(saving_percent * 100), truck_selected, "{:.0f}%".format(vfr * 100), "{:.0f}%".format(wfr * 100), "{:.1f}".format(mix), order_qty, slog],
                    "Ideal State": ["{:.1f}".format(ideal_unit_cost), "N/A", "0", "N/A", "N/A", ideal_truck_type, "{:.0f}%".format(ideal_vfr * 100), "{:.0f}%".format(ideal_wfr * 100), "{:.1f}".format(ideal_mix), "N/A", "N/A"]
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

            if "实际采纳数量" in order_data.columns.tolist():
                if order_data["实际采纳数量"] is not np.nan:
                    result = pd.DataFrame(
                        {
                            "Index": ["Unit Cost (RMB/PT)", "Spending (RMB)", "Loss (RMB)", "Saving (RMB)", "Saving (%)",
                                      "Truck Type", "VFR", "WFR", "Heavy/Light Mix(CBM/Ton)", "Quantity (CS)",
                                      "SLOG Impact"],
                            "Before Shaping": ["{:.1f}".format(base_unit_cost), "{:.0f}".format(base_cost),
                                               "{:.0f}".format(base_loss), "N/A", "N/A", base_truck_selected,
                                               "{:.0f}%".format(base_vfr * 100), "{:.0f}%".format(base_wfr * 100),
                                               "{:.1f}".format(base_mix), initial_order_qty, base_slog],
                            "Proposed Shaping": ["{:.1f}".format(unit_cost), "{:.0f}".format(cost), "{:.0f}".format(loss),
                                                 "{:.0f}".format(saving), "{:.0f}%".format(saving_percent * 100),
                                                 truck_selected, "{:.0f}%".format(vfr * 100), "{:.0f}%".format(wfr * 100),
                                                 "{:.1f}".format(mix), order_qty, slog],
                            "Adopt Shaping": ["{:.1f}".format(adopt_unit_cost), "{:.0f}".format(adopt_cost),
                                              "{:.0f}".format(adopt_loss), "N/A", "N/A", adopt_truck_selected,
                                              "{:.0f}%".format(adopt_vfr * 100), "{:.0f}%".format(adopt_wfr * 100),
                                              "{:.1f}".format(adopt_mix), adopt_order_qty, adopt_slog],
                            "Ideal State": ["{:.1f}".format(ideal_unit_cost), "N/A", "0", "N/A", "N/A", ideal_truck_type,
                                            "{:.0f}%".format(ideal_vfr * 100), "{:.0f}%".format(ideal_wfr * 100),
                                            "{:.1f}".format(ideal_mix), "N/A", "N/A"]
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
                                "headerName": "Adopt Shaping",
                                "field": "Adopt Shaping",
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
        label += 1

    st.divider()
    st.info("Recommendation Data Download")
    selected_order_data["建议调整数量"] = 0
    not_selected_order_data["建议调整数量"] = np.nan

    if order_data_result.empty:
        st.warning("**Warning:**" + "There is no order data within the testing scope in the file(s).")
    else:
        for material_num in order_data_result["material_num"]:
            order_data_related = selected_order_data[selected_order_data["material_num"] == material_num]
            if len(order_data_related) == 1:
                selected_order_data.loc[selected_order_data["material_num"] == material_num, "建议调整数量"] = list(order_data_result[order_data_result["material_num"] == material_num]["filler_qty"])
            else:
                qty_list = list(order_data_related["采购需求数量*"])
                qty_sum = order_data_related["采购需求数量*"].sum()
                total_qty = order_data_result[order_data_result["material_num"] == material_num]["filler_qty"]
                final_qty = []
                current_sum = 0
                for i in range(len(order_data_related)):
                    if i < len(order_data_related) - 1:
                        final_qty.append(round(total_qty / qty_sum * qty_list[i]))
                        current_sum += round(total_qty / qty_sum * qty_list[i])
                    else:
                        final_qty.append(total_qty - current_sum)
                selected_order_data.loc[selected_order_data["material_num"] == material_num, "建议调整数量"] = final_qty

        selected_order_data["建议调整数量"] = selected_order_data["建议调整数量"] * selected_order_data["箱规⑥"]

        for source in upload_source_list:
            if source in download_source_list:
                source_selected_order_data = selected_order_data[selected_order_data["Source"] == source]
                source_not_selected_order_data = not_selected_order_data[not_selected_order_data["Source"] == source]
                output_data = pd.concat([source_selected_order_data, source_not_selected_order_data])
                index = output_data.columns.tolist().index("采购需求数量*")
                if "横版" in source:
                    output_data.insert(index + 1, "北京_建议调整数量", output_data.pop("建议调整数量"))
                    if "北京_实际采纳数量" not in output_data.columns.tolist():
                        output_data.insert(index + 2, "北京_实际采纳数量", np.nan)
                    output_data = output_data.rename(columns={"京东码": "商品编码", "采购需求数量*": "北京"})
                    output_data.drop(["配送中心*(格式：北京,上海,广州)", "material_num", "箱规⑥", "CS", "max_filler_CS", "category", "volume_cube", "weight_ton", "Region", "shipto", "Source"], axis=1, inplace=True)
                    source = source.rstrip("_横版")
                else:
                    index = output_data.columns.tolist().index("采购需求数量*")
                    output_data.insert(index + 1, "建议调整数量", output_data.pop("建议调整数量"))
                    if "实际采纳数量" not in output_data.columns.tolist():
                        output_data.insert(index + 2, "实际采纳数量", np.nan)
                    output_data = output_data.rename(columns={"京东码": "sku*"})
                    output_data.drop(["material_num", "箱规⑥", "CS", "max_filler_CS", "category", "volume_cube", "weight_ton", "Region", "shipto", "Source"], axis=1, inplace=True)
                    source = source.rstrip("_竖版")

                output = BytesIO()
                with pd.ExcelWriter(output) as writer:
                    output_data.to_excel(writer, index=False)
                st.download_button(
                    label="Download Optimization Result for " + source,
                    data=output.getvalue(),
                    file_name=source.split(".")[0] + "_result.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:
                st.warning("**Warning:**" + "There is no order data within the testing scope in the file " + source)

