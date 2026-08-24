# ml_engine/dl_model.py

def dl_predict_logic(monthly_charges):
    """
    Simulates a Neural Network inference.
    In a real scenario, you would load a .h5 or .pth file here.
    """
    # Logic: High Monthly Charges + specific patterns trigger a DL '1'
    return 1 if monthly_charges > 89.9 else 0