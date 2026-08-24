def explain_prediction(row):
    explanation = []

    if row["Torque [Nm]"] > 65:
        explanation.append("High Torque")
    if row["Tool wear [min]"] > 40:
        explanation.append("High Tool Wear")
    if row["Air temperature [K]"] > 300:
        explanation.append("High Air Temperature")

    if not explanation:
        return "Normal operating conditions"
    
    return ", ".join(explanation)