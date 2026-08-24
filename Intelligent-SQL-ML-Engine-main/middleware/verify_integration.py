from middleware import intelligent_query_processor

# This query is designed to be "Heavy" to trigger the Optimizer 
# and contains the "PREDICT" keyword to trigger the ML Injection.
test_query = """
SELECT customer_name, PREDICT(churn_risk) 
FROM customers 
WHERE tenure > 12 AND MonthlyCharges > 50 AND region = 'North' OR status = 'Priority'
"""

print("🚀 RUNNING INTEGRATION TEST...\n")
result = intelligent_query_processor(test_query)

# --- VERIFY UPGRADE 1: Cost-Based Optimizer ---
print("--- [CHECK 1: OPTIMIZER] ---")
print(f"Complexity Score: {result['metadata']['complexity_score']}")
print(f"Calculated Cost: {result['metadata']['calculated_cost']}")
print(f"Selected Strategy: {result['strategy']}")

# --- VERIFY UPGRADE 2: Multi-Prediction Injection ---
print("\n--- [CHECK 2: ML INJECTION] ---")
print("Augmented SQL Output:")
print(result['augmented_sql'])

# --- FINAL VERDICT ---
print("\n--- FINAL VERDICT ---")
if "Risk_Level" in result['augmented_sql'] and result['metadata']['calculated_cost'] > 750:
    print("✅ SUCCESS: Optimizer and ML Injection are working together!")
else:
    print("❌ FAILED: Check if logic is correctly placed in middleware.py")