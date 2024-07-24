from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define a structure with edges: Parent node --> Child node
model = BayesianModel([('Network Protocol Usage Risk', 'Reliability'),
                       ('Network Protocol Usage Risk', 'Energy Efficiency'),
                       ('Data Exchange Volume Risk', 'Reliability'),
                       ('Data Exchange Volume Risk', 'Energy Efficiency')])

cpd_npur = TabularCPD(variable='Network Protocol Usage Risk', variable_card=2, values=[[0.50], [1-0.50]])
cpd_devr = TabularCPD(variable='Data Exchange Volume Risk', variable_card=2, values=[[0.50815], [1-0.50815]])

cpd_rel = TabularCPD(variable='Reliability', variable_card=2,
                     values=[[0.2, 0.5, 0.4, 0.7],
                             [0.8, 0.5, 0.6, 0.3]],
                     evidence=['Network Protocol Usage Risk', 'Data Exchange Volume Risk'],
                     evidence_card=[2, 2])

cpd_ee = TabularCPD(variable='Energy Efficiency', variable_card=2,
                    values=[[0.1, 0.3, 0.4, 0.6],
                            [0.9, 0.7, 0.6, 0.4]],
                    evidence=['Network Protocol Usage Risk', 'Data Exchange Volume Risk'],
                    evidence_card=[2, 2])

# Step 2: Add the CPDs to the network
model.add_cpds(cpd_npur, cpd_devr, cpd_rel, cpd_ee)

# Step 3: Verify the model
if model.check_model():
    print("Bayesian Network model is valid")
else:
    print("Bayesian Network model is NOT valid")

# Step 4: Make inferences
infer = VariableElimination(model)

# Print the results of the query
print("Reliability Query:")
print(infer.query(variables=['Reliability'], evidence={}).values)
print("Energy Efficiency Query:")
print(infer.query(['Energy Efficiency'], evidence={}).values)

# Step 5: Perform risk assessment via sensitivity analysis
print("\nPerforming Sensitivity Analysis for Risk Assessment:")
for i in range(2):
    for j in range(2):
        print(f"\nWhen Network Protocol Usage Risk is in state {i} and Data Exchange Volume Risk is in state {j}:")
        print(
            f"Reliability = {infer.query(['Reliability'], evidence={'Network Protocol Usage Risk': i, 'Data Exchange Volume Risk': j}).values}")
        print(
            f"Energy Efficiency = {infer.query(['Energy Efficiency'], evidence={'Network Protocol Usage Risk': i, 'Data Exchange Volume Risk': j}).values}")
