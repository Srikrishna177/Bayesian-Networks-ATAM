from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D


class BayesianNode:
    def __init__(self, name, states):
        self.name = name
        self.states = states
        self.parents = []  # List of parent nodes
        self.cpt = {}  # Conditional Probability Table (dictionary)

    def add_parent(self, parent):
        self.parents.append(parent)

    def set_cpt(self, cpt):
        self.cpt = cpt

    def get_cpt(self):
        return self.cpt

    def set_prob(self, parent_state_combination, state_probability):
        if isinstance(parent_state_combination, tuple) and isinstance(state_probability, dict):
            self.cpt[parent_state_combination] = state_probability
        else:
            raise ValueError("Invalid input types for parent_state_combination or state_probability.")

    def get_prob(self, parent_state_combination, state):
        try:
            return self.cpt[parent_state_combination][state]
        except KeyError:
            raise KeyError("The specified parent state combination or state does not exist in the CPT.")


class BayesianNetworkWrapper:
    def __init__(self):
        self.nodes = []
        self.cpds = {}
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, parent, child):
        self.edges.append((parent, child))

    def add_cpds(self, *cpds):
        for cpd in cpds:
            self.cpds[cpd.variable] = cpd

    def check_model(self):
        # Check that all nodes have CPDs
        for node in self.nodes:
            if node.name not in self.cpds:
                raise ValueError(f"No CPD associated with node {node.name}")

        # Check that CPDs are correctly defined
        for node in self.nodes:
            node_cpd = self.cpds[node.name]
            node_card = node_cpd.variable_card
            if len(node_cpd.variables) > 1:
                # this node has parents
                parents = node_cpd.variables[1:]  # excluding the first element, which is node's own name
                parent_card_prods = np.prod([self.cpds[parent].variable_card for parent in parents])
                child_card = node_card * parent_card_prods
            else:
                child_card = node_card

            if child_card != node_cpd.values.size:
                raise ValueError(f"Cardinality for {node.name} does not match the one specified in its CPD.")

        print("Model is valid")

    def get_node_by_name(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None


def print_cpd(cpd, variable):
    print("\nCPD of", variable, ":")
    states = cpd.state_names[variable]
    for state, prob in zip(states, cpd.values):
        # .6f formatter is used for 6 decimal precision
        print(f'P({variable}={state}) = {prob:.6f}')

def plot_tradeoff(inference):
    # 3 states each for Presentation_Layer, Business Logic Layer, and Persistence Layer
    layer_states = range(3)

    # Prepare lists to store the results
    reliabilities = []
    efficiencies = []
    presentation_layer_states = []
    business_logic_layer_states = []
    persistence_layer_states = []

    # Loop over all combinations
    for pl_state, bll_state, per_state in itertools.product(layer_states, repeat=3):
        rel_result = inference.query(variables=['Reliability'],
                                     evidence={'Presentation Layer': pl_state,
                                               'Business Logic Layer': bll_state,
                                               'Persistence Layer': per_state})
        ee_result = inference.query(variables=['Energy Efficiency'],
                                    evidence={'Presentation Layer': pl_state,
                                              'Business Logic Layer': bll_state,
                                              'Persistence Layer': per_state})

        # Append the high state probability to the lists
        reliabilities.append(rel_result.values[1])
        efficiencies.append(ee_result.values[1])
        presentation_layer_states.append(pl_state)
        business_logic_layer_states.append(bll_state)
        persistence_layer_states.append(per_state)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    # Create a scatter plot
    scatter1 = ax.scatter(presentation_layer_states, business_logic_layer_states, persistence_layer_states,
                         c=reliabilities, cmap='viridis')

    ax.set_xlabel('Presentation Layer')
    ax.set_ylabel('Business Logic Layer')
    ax.set_zlabel('Persistence Layer')
    fig.colorbar(scatter1, ax=ax, label='Reliability')

    ax1 = fig.add_subplot(122, projection='3d')
    scatter2 = ax1.scatter(presentation_layer_states, business_logic_layer_states, persistence_layer_states, c=efficiencies, cmap='viridis')
    ax1.set_xlabel('Presentation Layer')
    ax1.set_ylabel('Business Logic Layer')
    ax1.set_zlabel('Persistence Layer')
    fig.colorbar(scatter2, ax=ax1, label='Energy Efficiency')
    ax1.set_title('Energy Efficiency')

    plt.show()  # show the plot

def sensitivity_analysis(inference):
    # Create lists to store the results
    layers = ['Presentation Layer', 'Business Logic Layer', 'Persistence Layer']
    layer_states = range(3)  # Assuming your layers have 3 different states

    for layer in layers:
        print(f"\nPerforming sensitivity analysis for {layer}:")
        for state in layer_states:
            evidence = {layer: state}
            # You may need to adjust this according to any specific requirements
            result_reliability = inference.query(variables=['Reliability'], evidence=evidence)
            result_efficiency = inference.query(variables=['Energy Efficiency'], evidence=evidence)

            print(f"When {layer} is in state {state}:")
            print(f"Reliability = {result_reliability.values}")  # corrected line
            print(f"Energy Efficiency = {result_efficiency.values}")  # corrected line

def main():
    network = BayesianNetworkWrapper()

    '''
    # Define nodes for Components for the component-based architecture
    Environment_Configuration_Component = BayesianNode("Environment Configuration Component", ["Low", "Medium", "High"])
    Communication_Component = BayesianNode("Communication Component", ["Low", "Medium", "High"])
    Adaptation_Component = BayesianNode("Adaptation Component", ["Low", "Medium", "High"])
    # Define nodes for Features of Client-Server Architecture
    Network_Protocol_Usage = BayesianNode("Network Protocol Usage", ["Low", "Medium", "High"])
    Data_Exchange_Volume = BayesianNode("Data Exchange Volume", ["Low", "Medium", "High"])
    '''
    # Define nodes for the layers in Layered Architecture
    Presentation_Layer = BayesianNode("Presentation Layer", ["Low", "Medium", "High"])
    Business_Logic_Layer = BayesianNode("Business Logic Layer", ["Low", "Medium", "High"])
    Persistence_Layer = BayesianNode("Persistence Layer", ["Low", "Medium", "High"])
    # Define nodes for Quality Attributes
    Reliability = BayesianNode("Reliability", ["Low", "High"])
    Energy_Efficiency = BayesianNode("Energy Efficiency", ["Low", "High"])


    # Add nodes to the network
    '''
    network.add_node(Network_Protocol_Usage)
    network.add_node(Data_Exchange_Volume)
    network.add_node(Environment_Configuration_Component)
    network.add_node(Communication_Component)
    network.add_node(Adaptation_Component)
    '''
    network.add_node(Presentation_Layer)
    network.add_node(Business_Logic_Layer)
    network.add_node(Persistence_Layer)
    network.add_node(Reliability)
    network.add_node(Energy_Efficiency)

    # Define dependencies
    '''
    network.add_edge(Environment_Configuration_Component, Reliability)
    network.add_edge(Communication_Component, Reliability)
    network.add_edge(Adaptation_Component, Reliability)
    network.add_edge(Environment_Configuration_Component, Energy_Efficiency)
    network.add_edge(Communication_Component, Energy_Efficiency)
    network.add_edge(Adaptation_Component, Energy_Efficiency)
    network.add_edge(Network_Protocol_Usage, Reliability)
    network.add_edge(Data_Exchange_Volume, Reliability)
    network.add_edge(Network_Protocol_Usage, Energy_Efficiency)
    network.add_edge(Data_Exchange_Volume, Energy_Efficiency)
    '''
    network.add_edge(Presentation_Layer, Reliability)
    network.add_edge(Business_Logic_Layer, Reliability)
    network.add_edge(Persistence_Layer, Reliability)
    network.add_edge(Presentation_Layer, Energy_Efficiency)
    network.add_edge(Business_Logic_Layer, Energy_Efficiency)
    network.add_edge(Persistence_Layer, Energy_Efficiency)

    # Define the CPDs for PresentationLayer, BusinessLayer, PersistenceLayer
    cpd_presentation_layer = TabularCPD(variable='Presentation Layer', variable_card=3, values=[[0.825], [0.125], [0.05]])

    cpd_business_layer = TabularCPD(variable='Business Logic Layer', variable_card=3, values=[[0.174], [0.157], [0.66]])

    cpd_persistence_layer = TabularCPD(variable='Persistence Layer', variable_card=3, values=[[0.075], [0.20], [0.725]])
    '''
    # Define the CPDs for Network Protocol Usage, Data Exchange Volume
    cpd_npu = TabularCPD(variable='Network Protocol Usage', variable_card=3, values=[[0.10], [0.20], [0.70]])

    cpd_dev = TabularCPD(variable='Data Exchange Volume', variable_card=3, values=[[0.20], [0.70], [0.10]])

    # Define the CPDs for Configuration Component, Communication Component, Adaptation Component
    cpd_configuration = TabularCPD(variable='Environment Configuration Component', variable_card=3,
                                   values=[[0.50], [0.25], [0.25]])

    cpd_communication = TabularCPD(variable='Communication Component', variable_card=3, values=[[0.10], [0.65], [0.25]])

    cpd_adaptation = TabularCPD(variable='Adaptation Component', variable_card=3, values=[[0.10], [0.60], [0.30]])

    # Define the CPD for Reliability
    cpd_reliability_cba = TabularCPD(variable='Reliability', variable_card=2,
                                     evidence=['Environment Configuration Component', 'Communication Component',
                                               'Adaptation Component'], evidence_card=[3, 3, 3], values=[
            [0.10, 0.20, 0.30, 0.30, 0.40, 0.50, 0.50, 0.60, 0.70, 0.40, 0.50, 0.60, 0.60, 0.70, 0.80, 0.70, 0.80, 0.85,
             0.70, 0.80, 0.90, 0.85, 0.90, 0.95, 0.90, 0.95, 0.98],  # High Reliability
            [0.90, 0.80, 0.70, 0.70, 0.60, 0.50, 0.50, 0.40, 0.30, 0.60, 0.50, 0.40, 0.40, 0.30, 0.20, 0.30, 0.20, 0.15,
             0.30, 0.20, 0.10, 0.15, 0.10, 0.05, 0.10, 0.05, 0.02]  # Low Reliability
        ])

    # Define the CPD for Energy Efficiency
    cpd_energy_efficiency_cba = TabularCPD(variable='Energy Efficiency', variable_card=2,
                                           evidence=['Environment Configuration Component', 'Communication Component',
                                                     'Adaptation Component'], evidence_card=[3, 3, 3], values=[
            [0.20, 0.30, 0.40, 0.40, 0.50, 0.60, 0.50, 0.60, 0.70, 0.50, 0.60, 0.70, 0.60, 0.70, 0.80, 0.70, 0.80, 0.85,
             0.70, 0.80, 0.90, 0.85, 0.90, 0.95, 0.90, 0.95, 0.98],  # High Energy Efficiency
            [0.80, 0.70, 0.60, 0.60, 0.50, 0.40, 0.50, 0.40, 0.30, 0.50, 0.40, 0.30, 0.40, 0.30, 0.20, 0.30, 0.20, 0.15,
             0.30, 0.20, 0.10, 0.15, 0.10, 0.05, 0.10, 0.05, 0.02]  # Low Energy Efficiency
        ])

    # Define the CPD for Energy Efficiency
    cpd_ee_csa = TabularCPD(variable='Energy Efficiency', variable_card=2,
                            evidence=['Network Protocol Usage', 'Data Exchange Volume'], evidence_card=[3, 3], values=[
            [0.6, 0.7, 0.8, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5],  # EE_L
            [0.4, 0.3, 0.2, 0.6, 0.5, 0.4, 0.7, 0.6, 0.5]  # EE_H
        ])

    # Define the CPD for Reliability
    cpd_reliability_csa = TabularCPD(variable='Reliability', variable_card=2,
                                     evidence=['Network Protocol Usage', 'Data Exchange Volume'], evidence_card=[3, 3],
                                     values=[
                                         [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.05],
                                         [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 0.95]
                                     ])
    '''
    '''
    # Define the CPD for Reliability
    cpd_reliability_la = TabularCPD(variable='Reliability', variable_card=2, evidence=['Presentation Layer', 'Business Logic Layer', 'Persistence Layer'], evidence_card=[3, 3, 3], values=[
        [0.8, 0.7, 0.6, 0.7, 0.6, 0.5, 0.6, 0.5, 0.4, 0.7, 0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.4, 0.3, 0.6, 0.5, 0.4, 0.5, 0.4, 0.3, 0.4, 0.3, 0.2],  # Low Reliability
        [0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.5, 0.6, 0.7, 0.6, 0.7, 0.8]   # High Reliability
    ])

    # Define the CPD for Energy Efficiency
    cpd_energy_efficiency_la = TabularCPD(variable='Energy Efficiency', variable_card=2, evidence=['Presentation Layer', 'Business Logic Layer', 'Persistence Layer'], evidence_card=[3, 3, 3], values=[
        [0.7, 0.6, 0.5, 0.65, 0.5, 0.4, 0.5, 0.4, 0.3, 0.6, 0.5, 0.4, 0.5, 0.35, 0.3, 0.2, 0.5, 0.4, 0.3, 0.4, 0.3, 0.2, 0.4, 0.2, 0.3, 0.2, 0.1],  # Low Energy Efficiency
        [0.3, 0.4, 0.5, 0.35, 0.5, 0.6, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.5, 0.65, 0.7, 0.8, 0.5, 0.6, 0.7, 0.6, 0.7, 0.8, 0.6, 0.8, 0.7, 0.8, 0.9]   # High Energy Efficiency
    ])
    '''

    # Define the CPD for Reliability
    cpd_reliability_la = TabularCPD(variable='Reliability', variable_card=2,
                                    evidence=['Presentation Layer', 'Business Logic Layer', 'Persistence Layer'],
                                    evidence_card=[3, 3, 3], values=[
            [0.8, 0.75, 0.7, 0.75, 0.7, 0.65, 0.65, 0.6, 0.55, 0.75, 0.7, 0.65, 0.7, 0.65, 0.6, 0.6, 0.55, 0.5, 0.65, 0.6, 0.55, 0.6, 0.55, 0.5, 0.5, 0.45, 0.4],  # Low Reliability
            [0.2, 0.25, 0.3, 0.25, 0.3, 0.35, 0.35, 0.4, 0.45, 0.25, 0.3, 0.35, 0.3, 0.35, 0.4, 0.4, 0.45, 0.5, 0.35, 0.4, 0.45, 0.4, 0.45, 0.5, 0.5, 0.55, 0.6]  # High Reliability
        ])

    # Define the CPD for Energy Efficiency
    cpd_energy_efficiency_la = TabularCPD(variable='Energy Efficiency', variable_card=2,
                                          evidence=['Presentation Layer', 'Business Logic Layer', 'Persistence Layer'],
                                          evidence_card=[3, 3, 3], values=[
            [0.3, 0.35, 0.4, 0.35, 0.4, 0.45, 0.4, 0.45, 0.5, 0.35, 0.4, 0.45, 0.4, 0.45, 0.5, 0.45, 0.5, 0.55, 0.4, 0.45, 0.5, 0.45, 0.5, 0.55, 0.5, 0.55, 0.6],  # Low Energy Efficiency
            [0.7, 0.65, 0.6, 0.65, 0.6, 0.55, 0.6, 0.55, 0.5, 0.65, 0.6, 0.55, 0.6, 0.55, 0.5, 0.55, 0.5, 0.45, 0.6, 0.55, 0.5, 0.55, 0.5, 0.45, 0.5, 0.45, 0.4]  # High Energy Efficiency
        ])

    # Add CPDs to the model
    network.add_cpds(cpd_presentation_layer, cpd_business_layer, cpd_persistence_layer, cpd_energy_efficiency_la,
                     cpd_reliability_la)

    # Verify the model
    network.check_model()

    # Create the pgmpy model from the BayesianNetworkWrapper
    pgmpy_model = BayesianNetwork()
    for node in network.nodes:
        pgmpy_model.add_node(node.name)
    for parent, child in network.edges:
        pgmpy_model.add_edge(parent.name, child.name)
    for cpd in network.cpds.values():
        pgmpy_model.add_cpds(cpd)

    # Perform inference
    inference = VariableElimination(pgmpy_model)

    # Calculate the CPD of energy efficiency and reliability
    '''
    result_ee_csa = inference.query(variables=['Energy Efficiency'], joint=False)
    result_reliability_csa = inference.query(variables=['Reliability'], joint=False)
    result_ee_cba = inference.query(variables=['Energy Efficiency'],
                                    evidence={'Environment Configuration Component': 2, 'Communication Component': 2,
                                              'Adaptation Component': 2})
    result_reliability_cba = inference.query(variables=['Reliability'],
                                             evidence={'Environment Configuration Component': 2,
                                                       'Communication Component': 2, 'Adaptation Component': 2})
    '''
    result_reliability_la = inference.query(variables=['Reliability'],
                                            evidence={'Presentation Layer': 2, 'Business Logic Layer': 2,
                                                      'Persistence Layer': 2})
    result_ee_la = inference.query(variables=['Energy Efficiency'],
                                   evidence={'Presentation Layer': 2, 'Business Logic Layer': 2,
                                             'Persistence Layer': 2})

    print_cpd(result_reliability_la, 'Reliability')
    print_cpd(result_ee_la, 'Energy Efficiency')
    '''
    print_cpd(result_ee_csa, 'Energy Efficiency')
    print_cpd(result_reliability_csa, 'Reliability')
    print_cpd(result_ee_cba, 'Energy Efficiency')
    print_cpd(result_reliability_cba, 'Reliability')
    '''

    plot_tradeoff(inference)

    sensitivity_analysis(inference=inference)

if __name__ == "__main__":
    main()
