import streamlit as st
import pyodbc
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
truck_data = pd.read_excel("C:\\Users\\wang.n.22\\OneDrive - Procter and Gamble\\Documents\\Projects\\DPS\\Input_MD\\Tariff.xlsx", dtype={"Ship-to": str, "Truck Type": str, "Optimal Truck Type": str, "Customer Name": str})
sku_transfer_data = pd.read_excel("C:\\Users\\wang.n.22\\OneDrive - Procter and Gamble\\Documents\\Projects\\DPS\\Input_MD\\京东直供数据2024_0912.xlsx", dtype={"京东码": str, "宝洁码": str})
shipto_city_data = pd.read_excel("C:\\Users\\wang.n.22\\OneDrive - Procter and Gamble\\Documents\\Projects\\DPS\\Input_MD\\JD B2C 线路明细.xlsx", dtype={"City": str, "品类": str, "shipto": str})


@st.cache_data
def load_cdl_data(query):
    conn = pyodbc.connect("DSN=MyAzureDatabricks_DSN", autocommit=True)
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    sku_master = pd.DataFrame([list(row) for row in data], columns=columnNames)
    cursor.close()
    conn.close()
    return sku_master


query = """
    select right(MARM.material_num, 8) as material_num,
           (case fpc.category_en
                when "Baby" then "BC"
                when "Fabric" then "FHC"
                when "Home" then "FHC"
                when "Hair" then "HC"
                when "Oral" then "OC"
                when "Skin" then "SC"
                else fpc.category_en
           end) as category,
           (case MARM.volume_unit
                when "M3" then MARM.volume
                when "L" then MARM.volume / 1000
                when "DM3" then MARM.volume / 1000
                when "CM3" then MARM.volume / 1000000
           end) as volume_cube,
           (case MARM.weight_unit
                when "T" then MARM.gross_weight
                when "KG" then MARM.gross_weight / 1000
                when "G" then MARM.gross_weight / 1000000
           end) as weight_ton
    from dwd.tb_sc_mdm_prod_material_uom_dim as MARM
    left join dwd.tb_mdm_prod_fpc_dim as fpc
        on right(MARM.material_num, 8) = fpc.fpc_code
        and fpc.dw_is_current_flag = 1
    where alter_uom_for_sku = "CS"
    """
sku_master = load_cdl_data(query)


# File Uploading
uploaded_files = st.file_uploader(label="Please upload the order and truck type data file:", type="xlsx", accept_multiple_files=True)
order_data = pd.DataFrame()
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        original_order_data_split = pd.read_excel(uploaded_file, dtype={"sku*": str, "配送中心*(格式：北京,上海,广州)": str})
        order_data_split = original_order_data_split.rename(columns={"sku*": "京东码"})
        order_data_split = pd.merge(order_data_split, sku_transfer_data.loc[:, ["京东码", "宝洁码", "箱规⑥"]], how="left", on="京东码")
        order_data_split["CS"] = order_data_split["采购需求数量*"] / order_data_split["箱规⑥"]
        order_data_split["max_filler_CS"] = order_data_split["Max调整数量"] / order_data_split["箱规⑥"]
        order_data_split = order_data_split.rename(columns={"宝洁码": "material_num"})
        order_data_split = pd.merge(order_data_split, sku_master.loc[:, ["material_num", "category", "volume_cube", "weight_ton"]], how="left", on="material_num")
        order_data_split["City"] = order_data_split["配送中心*(格式：北京,上海,广州)"] + "市"
        category = np.array(order_data_split[order_data_split["category"].notnull()]["category"])
        shipto_city_data = shipto_city_data[shipto_city_data["品类"].str.contains(category[0], na=False)]
        order_data_split = pd.merge(order_data_split, shipto_city_data.loc[:, ["City", "shipto"]], how="left", on="City")
        order_data = pd.concat([order_data, order_data_split])
    order_data = order_data[order_data["shipto"].isin(["2003213268", "2002921387"])]
    order_data = pd.concat([order_data.groupby(by=["shipto", "material_num"])[["CS", "Max调整数量", "max_filler_CS"]].sum(), order_data.groupby(by=["shipto", "material_num"])[["category", "volume_cube", "weight_ton"]].max()], axis=1)
    order_data["category"] = order_data["category"].astype(str)
    order_data = order_data.rename(columns={"volume_cube": "unit_volume_cube", "weight_ton": "unit_weight_ton"})
    order_data["unit_volume_cube"] = order_data["unit_volume_cube"].astype(float)
    order_data["unit_weight_ton"] = order_data["unit_weight_ton"].astype(float)
    order_data["volume_cube"] = order_data["CS"] * order_data["unit_volume_cube"]
    order_data["weight_ton"] = order_data["CS"] * order_data["unit_weight_ton"]
    order_data = order_data.reset_index(names=["shipto", "material_num"])
    order_data["shipto"] = order_data["shipto"].astype(str)
    order_data["material_num"] = order_data["material_num"].astype(str)


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
    loss = (72.8 - unit_cost) * max(truck_loading_weight, truck_loading_volume / 3)
    wfr = truck_loading_weight / total_capacity_weight
    vfr = truck_loading_volume / total_capacity_volume
    mix = truck_loading_volume / truck_loading_weight

    return truck_qty, cost, unit_cost, loss, wfr, vfr, mix


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
    model.setParam(grb.GRB.Param.ObjNumber, 0)
    unit_cost = model.objNVal
    cost = unit_cost * max(truck_loading_weight, truck_loading_volume / 3)
    loss = (72.8 - unit_cost) * max(truck_loading_weight, truck_loading_volume / 3)
    wfr = truck_loading_weight / total_capacity_weight
    vfr = truck_loading_volume / total_capacity_volume
    mix = truck_loading_volume / truck_loading_weight

    return filler_qty, truck_qty, unit_cost, cost, loss, wfr, vfr, mix


def order_shaping(order_data, truck_data):
    order_data = order_data[order_data["weight_ton"].notnull()]
    order_data = order_data[order_data["volume_cube"].notnull()]

    # Parameters
    initial_order_weight = order_data["weight_ton"].sum()
    initial_order_volume = order_data["volume_cube"].sum()

    if len(order_data["material_num"]) >= 80:
        order_data = order_data.nlargest(80, "max_filler_CS")

    order_unit_weight = np.array(order_data["unit_weight_ton"])
    order_unit_volume = np.array(order_data["unit_volume_cube"])
    max_qty = np.array(order_data["max_filler_CS"])
    min_qty = - max_qty

    if "Priority" in list(order_data.columns):
        priority_param = np.array(order_data["Priority"])
    else:
        priority_param = np.zeros(len(order_data["material_num"]))

    truck_capacity_weight = np.array(truck_data["Weight Capacity"])
    truck_capacity_volume = np.array(truck_data["Volume Capacity"])
    truck_cost = np.array(truck_data["Base Charge"])

    ideal_truck_type = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Truck Type"])
    ideal_truck_capacity_weight = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Weight Capacity"])
    ideal_truck_capacity_volume = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Volume Capacity"])
    ideal_truck_cost = np.array(truck_data[truck_data["Optimal Truck Type"] == "Y"]["Base Charge"])

    # Calculate
    base_truck_qty, base_cost, base_unit_cost, base_loss, base_wfr, base_vfr, base_mix = baseline(initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume, truck_cost)
    filler_qty, truck_qty, unit_cost, cost, loss, wfr, vfr, mix = model_execution(initial_order_weight, initial_order_volume,
                                                                 order_unit_weight,
                                                                 order_unit_volume, max_qty, min_qty,
                                                                 priority_param, ideal_truck_capacity_weight,
                                                                 ideal_truck_capacity_volume, ideal_truck_cost)

    return base_truck_qty, base_cost, base_unit_cost, base_loss, base_wfr, base_vfr, base_mix, filler_qty, truck_qty, unit_cost, cost, loss, wfr, vfr, mix


# Calculation Execution and Display Result
if uploaded_files is not None:
    for shipto in np.array(order_data["shipto"].unique()):
        order_data = order_data[order_data["shipto"] == shipto]
        truck_data = truck_data[truck_data["Ship-to"] == shipto]
        initial_order_qty = order_data["CS"].sum()
        truck_type = np.array(truck_data["Truck Type"])

        base_truck_qty, base_cost, base_unit_cost, base_loss, base_wfr, base_vfr, base_mix, filler_qty, truck_qty, unit_cost, cost, loss, wfr, vfr, mix = order_shaping(order_data, truck_data)
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

        saving = base_cost - cost
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
                    truck_selected += str(truck_type[i]) + "*" + "{:.0f}".format(truck_qty[i])
                else:
                    truck_selected += "+" + str(truck_type[i]) + "*" + "{:.0f}".format(truck_qty[i])
        if order_qty >= 3500:
            slog = "2.3%"
        elif order_qty >= 2000:
            slog = "2.1%"
        elif order_qty >= 800:
            slog = "1.5%"
        else:
            slog = "0%"

        st.info("Basic Info")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Ship-to")
                st.markdown("Customer")
            with col2:
                st.markdown("2002921387")
                st.markdown("北京京东世纪贸易有限公司")

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
                "Ideal State": ["72.8", "N/A", "0", "N/A", "N/A", "9.6GL", "91%", "100%", "2.6", "N/A", "N/A"]
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
                "Category": order_data["category"],
                "SKU Number": order_data["material_num"],
                "SKU Quantity Limit (-MIN CS, +MAX CS)": ["(" + str(min_qty) + ", +" + str(max_qty) + ")" for min_qty, max_qty in zip(np.array(- order_data["Max调整数量"]), np.array(order_data["Max调整数量"]))],
                #"Reco Quantity (CS)": np.where(filler_qty <= 0, filler_qty, ["+" + str(qty) for qty in filler_qty])
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
