from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network for all components
model = BayesianModel([
    ('Configuration_Component_Risk', 'Reliability'),
    ('Configuration_Component_Risk', 'Energy_Efficiency'),
    ('Communication_Component_Risk', 'Reliability'),
    ('Communication_Component_Risk', 'Energy_Efficiency'),
    ('Adaptation_Component_Risk', 'Reliability'),
    ('Adaptation_Component_Risk', 'Energy_Efficiency'),
])

# Define CPDs for each component's risk
cpd_Adaptation_Component_Risk = TabularCPD(variable='Adaptation_Component_Risk', variable_card=2, values=[[1-0.5458], [0.5458]])
cpd_Communication_Component_Risk = TabularCPD(variable='Communication_Component_Risk', variable_card=2, values=[[1-0.3231], [0.3231]])
cpd_Configuration_Component_Risk = TabularCPD(variable='Configuration_Component_Risk', variable_card=2, values=[[1-0.5672], [0.5672]])

# Define CPDs for Reliability based on the combined effect of all component risks
cpd_Reliability = TabularCPD(variable='Reliability', variable_card=2,
                             values=[
                                 # Reliability = Satisfied
                                 [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                                 # Reliability = Not Satisfied
                                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                             evidence=['Adaptation_Component_Risk', 'Communication_Component_Risk', 'Configuration_Component_Risk'],
                             evidence_card=[2, 2, 2])

# Define CPDs for Energy Efficiency based on the combined effect of all component risks
cpd_Energy_Efficiency = TabularCPD(variable='Energy_Efficiency', variable_card=2,
                                   values=[
                                       # Energy_Efficiency = Satisfied
                                       [0.5, 0.4, 0.6, 0.3, 0.7, 0.8, 0.9, 0.9],
                                       # Energy_Efficiency = Not Satisfied
                                       [0.5, 0.6, 0.4, 0.7, 0.3, 0.2, 0.1, 0.1]],
                                   evidence=['Adaptation_Component_Risk', 'Communication_Component_Risk', 'Configuration_Component_Risk'],
                                   evidence_card=[2, 2, 2])

# Step 2 : Add CPDs to the model
model.add_cpds(cpd_Adaptation_Component_Risk, cpd_Communication_Component_Risk, cpd_Configuration_Component_Risk,
               cpd_Reliability, cpd_Energy_Efficiency)

# Step 3 : Verify the model
if model.check_model():
    print("Bayesian Network model is valid")
else:
    print("Bayesian Network model is NOT valid")

# Perform inference
inference = VariableElimination(model)

# Query for Reliability
query_result_reliability = inference.query(variables=['Reliability'],
                                           evidence={}).values
print("Reliability:", query_result_reliability)

# Query for Energy Efficiency
query_result_energy_efficiency = inference.query(variables=['Energy_Efficiency'],
                                                 evidence={}).values
print("Energy Efficiency:", query_result_energy_efficiency)

# Step 5: Perform risk assessment via sensitivity analysis

print("\nPerforming Sensitivity Analysis for Risk Assessment:")
for i in range(2):
    for j in range(2):
        for k in range(2):
         print(f"\nWhen Configuration_Component_Risk is in state {i} and Communication_Component_Risk is in state {j} and Adaptation_Component_Risk is in state {k}:")
         print(
            f"Reliability = {inference.query(['Reliability'], evidence={'Configuration_Component_Risk': i, 'Communication_Component_Risk': j, 'Adaptation_Component_Risk': k}).values}")
         print(
            f"Energy Efficiency = {inference.query(['Energy_Efficiency'], evidence={'Configuration_Component_Risk': i, 'Communication_Component_Risk': j, 'Adaptation_Component_Risk': k}).values}")

'''
# Records the maximum change in variables for each component when its state is flipped
components = ['Configuration_Component_Risk', 'Communication_Component_Risk', 'Adaptation_Component_Risk']
max_changes = {component: {'Reliability': 0, 'Energy_Efficiency': 0} for component in components}

# Defines a threshold for minor change
change_threshold = 0.1

# Conducts the sensitivity analysis
for i in range(2):
    for j in range(2):
        for k in range(2):
            evidence = {'Configuration_Component_Risk': i, 'Communication_Component_Risk': j,
                        'Adaptation_Component_Risk': k}

            reliability = inference.query(['Reliability'], evidence=evidence).values
            energy_efficiency = inference.query(['Energy_Efficiency'], evidence=evidence).values

            for component in components:
                # Flips the state of the component
                flipped_evidence = evidence.copy()
                flipped_evidence[component] = 1 - flipped_evidence[component]

                # Computes the change in variables when the component's state is flipped
                reliability_change = abs(
                    reliability - inference.query(['Reliability'], evidence=flipped_evidence).values).max()
                energy_efficiency_change = abs(
                    energy_efficiency - inference.query(['Energy_Efficiency'], evidence=flipped_evidence).values).max()

                # Updates the maximum change for the component
                max_changes[component]['Reliability'] = max(max_changes[component]['Reliability'], reliability_change)
                max_changes[component]['Energy_Efficiency'] = max(max_changes[component]['Energy_Efficiency'],
                                                                  energy_efficiency_change)

# Identifies the 'non-risk' components
non_risks = [component for component, changes in max_changes.items() if max(changes.values()) < change_threshold]
if non_risks:
    print(f"\nThe non-risk components are: {', '.join(non_risks)}")
else:
    print("No non-risk components found")
    '''