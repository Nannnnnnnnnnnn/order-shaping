import streamlit as st
import pandas as pd
import numpy as np
import gurobipy as grb


# Page and Title
st.set_page_config(layout="wide")
st.title("Order Shaping Tool")


# File Uploading
uploaded_file = st.file_uploader(label="Please upload the order and truck type data file:", type="xlsx")
if uploaded_file is not None:
    order_data = pd.read_excel(uploaded_file, sheet_name="订单汇总(中转VL06)",
                               dtype={"订单": str, "Destination Location ID": str, "Material": str, "category_en": str})
    truck_data = pd.read_excel(uploaded_file, sheet_name="车型数据")


# Calculation Function
def order_shaping(order_data, truck_data):
    # Parameters
    customer_order_weight = order_data["Weight (Ton)"][0]
    customer_order_volume = order_data["Volume (CU. M)"][0]

    intersite_order_weight = np.array(order_data["Weight (Ton)"][1:])
    intersite_order_volume = np.array(order_data["Volume (CU. M)"][1:])
    max_qty = np.array(order_data["Max Quantity"][1:])
    intersite_order_unit_weight = [weight / qty for weight, qty in zip(intersite_order_weight, max_qty)]
    intersite_order_unit_volume = [volume / qty for volume, qty in zip(intersite_order_volume, max_qty)]
    max_pt = max(order_data["Weight (Ton)"].sum(), order_data["Volume (CU. M)"].sum() / 3)
    min_pt = max(customer_order_weight, customer_order_volume / 3)

    priority_param = np.array(order_data["优先级"][1:])

    truck_type = np.array(truck_data["车型"])
    truck_capacity_weight = np.array(truck_data["载重"])
    truck_capacity_volume = np.array(truck_data["容积"])
    truck_cost = np.array(truck_data["Cost"])

    M = 100000

    destination = np.array(order_data["Destination Location ID"][1:])
    material = np.array(order_data["Material"][1:])

    # Construct Model Object
    model = grb.Model("VMI_Order")

    # Introduce Decision Variables
    n = len(intersite_order_weight)
    m = len(truck_type)
    l = n * 2 + m * 2 + 2
    p = model.addVar(lb=1 / max_pt, ub=1 / min_pt, vtype=grb.GRB.CONTINUOUS, name="p")
    q = [[] for i in range(l)]
    for i in range(l):
        if i in range(n):
            q[i] = model.addVar(lb=0, ub=max_qty[i] / min_pt, vtype=grb.GRB.CONTINUOUS, name="q_" + str(i))
        elif i in range(n, n + m):
            q[i] = model.addVar(lb=0, ub=float("inf"), vtype=grb.GRB.CONTINUOUS, name="q_" + str(i))
        elif i in range(n + m, n * 2 + m):
            q[i] = model.addVar(lb=0, ub=max_qty[i - n - m], vtype=grb.GRB.INTEGER, name="q_" + str(i))
        elif i in range(n * 2 + m, n * 2 + m * 2):
            q[i] = model.addVar(lb=0, ub=float("inf"), vtype=grb.GRB.INTEGER, name="q_" + str(i))
        else:
            q[i] = model.addVar(vtype=grb.GRB.BINARY, name="q_" + str(i))

    # Objective Function
    priority = grb.LinExpr(priority_param, q[:n])
    cost_per_pt = grb.LinExpr(truck_cost, q[n:(n + m)])
    model.setObjectiveN(cost_per_pt, index=0, priority=10, weight=1, name="Cost/PT")
    model.setObjectiveN(priority, index=4, priority=0, name="Priority")

    # Constraints
    model.addConstr(grb.LinExpr(intersite_order_unit_weight, q[:n]) + customer_order_weight * p <=
                    grb.LinExpr(truck_capacity_weight, q[n:(n + m)]), "Capacity Constraint")
    model.addConstr(grb.LinExpr(intersite_order_unit_volume, q[:n]) + customer_order_volume * p <=
                    grb.LinExpr(truck_capacity_volume, q[n:(n + m)]), "Capacity Constraint")
    model.addConstr(grb.LinExpr(intersite_order_unit_weight, q[:n]) + customer_order_weight * p >=
                    1 - M * (1 - q[n * 2 + m * 2]), "PT Constraint")
    model.addConstr((grb.LinExpr(intersite_order_unit_volume, q[:n]) + customer_order_volume * p) / 3 >=
                    1 - M * (1 - q[n * 2 + m * 2 + 1]), "PT Constraint")
    model.addConstr(q[n * 2 + m * 2] + q[n * 2 + m * 2 + 1] >= 1, "PT Constraint")
    for i in range(n):
        model.addQConstr(q[i] - p * q[n + m + i] == 0, "Integer Constraint")
    for i in range(m):
        model.addQConstr(q[n + i] - p * q[n * 2 + m + i] == 0, "Integer Constraint")

    # Solve the Constructed Model
    model.optimize()

    # Calculate Result
    total_capacity_weight = 0
    total_capacity_volume = 0
    truck_loading_weight = customer_order_weight
    truck_loading_volume = customer_order_volume
    cs_num = []
    truck_num = []

    for var in model.getVars():
        if model.status == grb.GRB.OPTIMAL:
            if var.index == 0:
                p_value = var.x
            if var.index in range(1, n + 1):
                cs_num.append(var.x / p_value)
                truck_loading_weight += var.x / p_value * intersite_order_unit_weight[var.index - 1]
                truck_loading_volume += var.x / p_value * intersite_order_unit_volume[var.index - 1]
            elif var.index in range(n + 1, n + m + 1):
                truck_num.append(var.x / p_value)
                total_capacity_weight += var.x / p_value * truck_capacity_weight[var.index - n - 1]
                total_capacity_volume += var.x / p_value * truck_capacity_volume[var.index - n - 1]
    order_shaping_result = pd.DataFrame({"Destination": destination, "Material": material, "CS #": cs_num})
    truck_planning_result = pd.DataFrame({"Truck Type": truck_type, "Truck #": truck_num})
    model.setParam(grb.GRB.Param.ObjNumber, 0)
    unit_cost = model.objNVal
    wfr = truck_loading_weight / total_capacity_weight
    vfr = truck_loading_volume / total_capacity_volume

    return order_shaping_result, truck_planning_result, unit_cost, wfr, vfr


# Calculation Execution
if uploaded_file is not None:
    order_shaping_result, truck_planning_result, unit_cost, wfr, vfr = order_shaping(order_data, truck_data)
    st.write(order_shaping_result)
    st.write(truck_planning_result)
    st.write("Cost/PT: ", unit_cost)
    st.write("WFR: ", wfr)
    st.write("VFR: ", vfr)
