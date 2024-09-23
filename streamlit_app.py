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
                                      "宝洁DC": str, "Sales Unit": str},
                               parse_dates=["预约到货时间"])
    truck_data = pd.read_excel(uploaded_file, dtype={"车型": str}, sheet_name="车型数据")


# Calculation Function
def baseline(initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume,
                    truck_cost):
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
    filler_qty = []
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
    model = grb.Model("VMI_Order")

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
    loss = (72.8 - unit_cost) * max(truck_loading_weight, truck_loading_volume / 3)
    wfr = truck_loading_weight / total_capacity_weight
    vfr = truck_loading_volume / total_capacity_volume
    mix = truck_loading_volume / truck_loading_weight

    return filler_qty, truck_qty, unit_cost, loss, wfr, vfr, mix


def order_shaping(order_data, truck_data):
    # Parameters
    initial_order_weight = order_data["Initial  Order Weight/Ton"].sum()
    initial_order_volume = order_data["Initial  Order Volume/M3"].sum()

    order_unit_weight = np.array(order_data["Weight/KG/CS"]) / 1000
    order_unit_volume = np.array(order_data["VolumeDM3/CS"]) / 1000
    max_qty = np.array(order_data["Max到货（箱数）"])
    min_qty = np.array(order_data["Min到货（箱数）"])

    priority_param = np.array(order_data["Priority"])

    truck_capacity_weight = np.array(truck_data["载重"])
    truck_capacity_volume = np.array(truck_data["容积"])
    truck_cost = np.array(truck_data["Cost"])

    # Calculate
    base_truck_qty, base_cost, base_unit_cost, base_loss, base_wfr, base_vfr, base_mix = baseline(initial_order_weight, initial_order_volume, truck_capacity_weight, truck_capacity_volume,
             truck_cost)
    filler_qty, truck_qty, unit_cost, loss, wfr, vfr, mix = model_execution(initial_order_weight, initial_order_volume,
                                                                 order_unit_weight,
                                                                 order_unit_volume, max_qty, min_qty,
                                                                 priority_param, truck_capacity_weight,
                                                                 truck_capacity_volume, truck_cost)

    return base_truck_qty, base_cost, base_unit_cost, base_loss, base_wfr, base_vfr, base_mix, filler_qty, truck_qty, unit_cost, loss, wfr, vfr, mix


# Calculation Execution
if uploaded_file is not None:
    initial_order_qty = order_data["Initial  Order/CS"].sum()
    destination = np.array(order_data["Ship to"])
    material = np.array(order_data["宝洁八位码"])

    truck_type = np.array(truck_data["车型"])
    truck_capacity_weight = np.array(truck_data["载重"])
    truck_capacity_volume = np.array(truck_data["容积"])
    truck_cost = np.array(truck_data["Cost"])

    base_truck_qty, base_cost, base_unit_cost, base_loss, base_wfr, base_vfr, base_mix, filler_qty, truck_qty, unit_cost, loss, wfr, vfr, mix = order_shaping(order_data, truck_data)
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

    filler_qty = np.array([round(qty) for qty in filler_qty])
    qty_changed = 0
    for i in range(len(filler_qty)):
        qty_changed += filler_qty[i]
        order_qty = initial_order_qty + qty_changed

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
            st.markdown("{:.0f}".format(loss - base_loss))
            st.markdown(qty_changed)

    result = pd.DataFrame(
        {
            "Index": ["Unit Cost (RMB/PT)", "Loss (RMB)", "Truck Type", "VFR", "WFR", "Heavy/Light Mix(CBM/Ton)", "Quantity (CS)", "SLOG Impact"],
            "Before Shaping": ["{:.1f}".format(base_unit_cost), "{:.0f}".format(base_loss), base_truck_selected, "{:.0f}%".format(base_vfr * 100), "{:.0f}%".format(base_wfr * 100), "{:.1f}".format(base_mix), initial_order_qty, base_slog],
            "After Shaping": ["{:.1f}".format(unit_cost), "{:.0f}".format(loss), truck_selected, "{:.0f}%".format(vfr * 100), "{:.0f}%".format(wfr * 100), "{:.1f}".format(mix), order_qty, slog],
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
            "SKU Quantity Limit (-MIN CS, +MAX CS)": ["(" + str(min_qty) + ", +" + str(max_qty) + ")" for min_qty, max_qty in zip(np.array(order_data["Min到货（箱数）"]), np.array(order_data["Max到货（箱数）"]))],
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
