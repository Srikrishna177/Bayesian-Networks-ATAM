import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network for all layers
model = BayesianNetwork([
    ('Presentation_Layer_Risk', 'Reliability'),
    ('Presentation_Layer_Risk', 'Energy_Efficiency'),
    ('Business_Layer_Risk', 'Reliability'),
    ('Business_Layer_Risk', 'Energy_Efficiency'),
    ('Persistence_Layer_Risk', 'Reliability'),
    ('Persistence_Layer_Risk', 'Energy_Efficiency')
])

# Combined probability for Presentation Layer
prob_presentation_layer_risk_occurs = 0.7258

# Combined probability for Business Layer
prob_business_layer_risk_occurs = 0.4061

# Combined probability for Persistence Layer
prob_persistence_layer_risk_occurs = 0.7385

# Define CPDs for Presentation Layer Risk
cpd_Presentation_Layer_Risk = TabularCPD(variable='Presentation_Layer_Risk', variable_card=2,
                                         values=[[prob_presentation_layer_risk_occurs], [1 - prob_presentation_layer_risk_occurs]])

# Define CPDs for Business Layer Risk based on Presentation Layer Risk
cpd_Business_Layer_Risk = TabularCPD(variable='Business_Layer_Risk', variable_card=2,
                                     values=[[prob_business_layer_risk_occurs], [1 - prob_business_layer_risk_occurs]])

# Define CPDs for Persistence Layer Risk based on Presentation Layer Risk
cpd_Persistence_Layer_Risk = TabularCPD(variable='Persistence_Layer_Risk', variable_card=2,
                                        values=[[prob_persistence_layer_risk_occurs], [1 - prob_persistence_layer_risk_occurs]])

# Define CPDs for quality attributes based on Business Layer Risk
cpd_Reliability = TabularCPD(variable='Reliability', variable_card=2,
                                      values=[[0.8, 0.4, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],  # Satisfied
                                              [0.2, 0.6, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],  # Not Satisfied
                                      evidence=['Business_Layer_Risk', 'Persistence_Layer_Risk', 'Presentation_Layer_Risk'],
                                      evidence_card=[2, 2, 2])

cpd_Energy_Efficiency = TabularCPD(variable='Energy_Efficiency', variable_card=2,
                                            values=[[0.9, 0.5, 0.6, 0.3, 0.7, 0.8, 0.9, 0.95],  # Satisfied
                                                    [0.1, 0.5, 0.4, 0.7, 0.3, 0.2, 0.1, 0.05]],  # Not Satisfied
                                            evidence=['Business_Layer_Risk', 'Persistence_Layer_Risk', 'Presentation_Layer_Risk'],
                                            evidence_card=[2, 2, 2])
'''
# Define CPDs for quality attributes based on Persistence Layer Risk
cpd_Reliability_Persistence = TabularCPD(variable='Reliability', variable_card=2,
                                         values=[[0.8, 0.4],  # Satisfied
                                                 [0.2, 0.6]],  # Not Satisfied
                                         evidence=['Persistence_Layer_Risk'],
                                         evidence_card=[2])

cpd_Energy_Efficiency_Persistence = TabularCPD(variable='Energy_Efficiency', variable_card=2,
                                               values=[[0.9, 0.5],  # Satisfied
                                                       [0.1, 0.5]],  # Not Satisfied
                                               evidence=['Persistence_Layer_Risk'],
                                               evidence_card=[2])
'''

# Step 2 : Add CPDs to the model
model.add_cpds(cpd_Presentation_Layer_Risk, cpd_Business_Layer_Risk, cpd_Persistence_Layer_Risk , cpd_Energy_Efficiency, cpd_Reliability)

# Step 3 : Verify the model
if model.check_model():
    print("Bayesian Network model is valid")
else:
    print("Bayesian Network model is NOT valid")

# Step 4 : Perform inference
inference = VariableElimination(model)

# Query for Reliability and Energy Efficiency
query_result_reliability = inference.query(variables=['Reliability'], evidence={}).values
print("Reliability:", query_result_reliability)
query_result_energy_efficiency = inference.query(variables=['Energy_Efficiency'], evidence={}).values
print("Energy Efficiency:", query_result_energy_efficiency)

print("\nPerforming Sensitivity Analysis for Risk Assessment:")
for i in range(2):
    for j in range(2):
        for k in range(2):
         print(f"\nWhen Presentation_Layer_Risk is in state {i} and Business_Layer_Risk is in state {j} and Persistence_Layer_Risk is in state {k}:")
         print(
            f"Reliability = {inference.query(['Reliability'], evidence={'Presentation_Layer_Risk': i, 'Business_Layer_Risk': j, 'Persistence_Layer_Risk': k}).values}")
         print(
            f"Energy Efficiency = {inference.query(['Energy_Efficiency'], evidence={'Presentation_Layer_Risk': i, 'Business_Layer_Risk': j, 'Persistence_Layer_Risk': k}).values}")

