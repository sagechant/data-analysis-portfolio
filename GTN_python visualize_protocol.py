import networkx as nx
from pyvis.network import Network

# 1. Define Nodes and their features (simplified for display)
#    We use node IDs, visible labels, and 'title' for hover information
nodes = [
    # Initialization
    {'id': 'S_Start', 'label': 'Start', 'type': 'Start', 'title': 'Protocol Start'},
    {'id': 'S_On_Comp', 'label': 'Turn On Computer', 'type': 'Step', 'title': 'Action: Turn on computer if needed'},
    {'id': 'N_Computer', 'label': 'Computer', 'type': 'Device', 'title': 'Device: Computer connected to NanoDrop'},
    {'id': 'S_Open_SW', 'label': 'Open Software', 'type': 'Software Action', 'title': 'Software Action: Open NanoDrop 2000c software'},
    {'id': 'N_Software', 'label': 'NanoDrop SW', 'type': 'Software', 'title': 'Software: NanoDrop 2000c'},
    {'id': 'SE_Desktop', 'label': 'Desktop', 'type': 'Software Element', 'title': 'Software Context: Desktop'},
    {'id': 'S_Wait_Init', 'label': 'Wait Init', 'type': 'Non-action', 'title': 'Non-action: Wait for instrument to initialize'},
    {'id': 'D_NanoDrop', 'label': 'NanoDrop', 'type': 'Device', 'title': 'Device: NanoDrop Spectrophotometer'},
    {'id': 'ST_Init_Done', 'label': 'Init Done', 'type': 'State', 'title': 'Device State: Instrument Initialized'},
    {'id': 'S_Select_Opt', 'label': 'Select Option', 'type': 'Software Action', 'title': 'Software Action: Select Nucleic Acid measurement option'},
    {'id': 'SE_NuclAcid_Opt', 'label': 'Nucl. Acid Option', 'type': 'Software Element', 'title': 'Software Element: Nucleic Acid measurement option'},

    # Blanking
    {'id': 'S_RaiseArm_1', 'label': 'Raise Arm (Blank)', 'type': 'Step', 'title': 'Action: Raise the NanoDrop sampling arm'},
    {'id': 'ST_Arm_Raised_1', 'label': 'Arm Raised 1', 'type': 'State', 'title': 'Device State: NanoDrop Arm Raised (Blanking)'},
    {'id': 'S_Pipette_Blank', 'label': 'Pipette Blank', 'type': 'Step', 'title': 'Action: Pipette blank buffer onto lower pedestal'},
    {'id': 'M_BlankBuffer', 'label': 'Blank Buffer', 'type': 'Material', 'title': 'Material: Blank buffer (1 µL used)'}, # Quantity in title
    {'id': 'D_Pipette', 'label': 'Pipette', 'type': 'Device', 'title': 'Device: 2 µL pipette'},
    {'id': 'D_PipetteTips', 'label': 'Pipette Tips', 'type': 'Device Accessory', 'title': 'Device Accessory: Pipette tips'},
    {'id': 'S_LowerArm_1', 'label': 'Lower Arm (Blank)', 'type': 'Step', 'title': 'Action: Lower the sampling arm'},
    {'id': 'ST_Arm_Lowered_1', 'label': 'Arm Lowered 1', 'type': 'State', 'title': 'Device State: NanoDrop Arm Lowered (Blanking)'},
    {'id': 'S_Click_Blank', 'label': 'Click Blank', 'type': 'Software Action', 'title': 'Software Action: Click Blank to zero instrument'},
    {'id': 'SE_Blank_Button', 'label': 'Blank Button', 'type': 'Software Element', 'title': 'Software Element: Blank Button'},
    {'id': 'ST_Blanked', 'label': 'Instrument Blanked', 'type': 'State', 'title': 'Device State: Instrument is zeroed/blanked'},

    # Loop Preparation (Before sample measurement loop)
    # Reusing RaiseArm_2/LowerArm_2 as they are distinct steps in the procedure flow
    {'id': 'S_RaiseArm_2', 'label': 'Raise Arm (Sample)', 'type': 'Step', 'title': 'Action: Raise the arm before sample measurement'},
    {'id': 'ST_Arm_Raised_2', 'label': 'Arm Raised 2', 'type': 'State', 'title': 'Device State: NanoDrop Arm Raised (Sample)'},
    {'id': 'S_Wipe_1', 'label': 'Wipe Pedestals 1', 'type': 'Step', 'title': 'Action: Wipe NanoDrop pedestals with a lint-free wipe'},
    {'id': 'M_Wipe', 'label': 'Lint-Free Wipe', 'type': 'Material', 'title': 'Material: Lint-free wipe'},
    {'id': 'S_Write_Label', 'label': 'Write Sample Label', 'type': 'Software Action', 'title': 'Software Action: Write sample label in software'},
    {'id': 'M_CurrentSample', 'label': 'Current DNA Sample', 'type': 'Material', 'title': 'Material: The current DNA sample (1 µL used)'}, # Quantity in title
    {'id': 'SE_Label_Field', 'label': 'Label Field', 'type': 'Software Element', 'title': 'Software Element: Sample Label Input Field'},

    # Inside the Measurement Loop
    {'id': 'S_Pipette_Sample', 'label': 'Pipette Sample', 'type': 'Step', 'title': 'Action: Pipette DNA sample onto lower pedestal'},
    {'id': 'S_LowerArm_2', 'label': 'Lower Arm (Sample)', 'type': 'Step', 'title': 'Action: Lower the sampling arm for measurement'},
    {'id': 'ST_Arm_Lowered_2', 'label': 'Arm Lowered 2', 'type': 'State', 'title': 'Device State: NanoDrop Arm Lowered (Sample)'},
    {'id': 'S_Click_Measure', 'label': 'Click Measure', 'type': 'Software Action', 'title': 'Software Action: Click Measure'},
    {'id': 'SE_Measure_Button', 'label': 'Measure Button', 'type': 'Software Element', 'title': 'Software Element: Measure Button'},
    {'id': 'S_Wait_Measure', 'label': 'Wait Measure', 'type': 'Non-action', 'title': 'Non-action: Wait until measurement is done'},
    {'id': 'ST_Meas_Done', 'label': 'Measurement Done', 'type': 'State', 'title': 'Process State: Measurement is complete'},
    {'id': 'S_Save_Meas', 'label': 'Save Measurement', 'type': 'Software Action', 'title': 'Software Action: Save the measurement'},
    {'id': 'M_MeasData', 'label': 'Measurement Data', 'type': 'Data Output', 'title': 'Data: Recorded Measurement (Concentration, Purity Ratios)'},
    {'id': 'N_DNA_Samples_Inv', 'label': 'DNA Samples (Inventory)', 'type': 'Material', 'title': 'Material: Source of all DNA samples'}, # Represents the collection

    # Cleanup
    {'id': 'S_RaiseArm_3', 'label': 'Raise Arm (Cleanup)', 'type': 'Step', 'title': 'Action: Raise the arm for cleanup'},
    {'id': 'ST_Arm_Raised_3', 'label': 'Arm Raised 3', 'type': 'State', 'title': 'Device State: NanoDrop Arm Raised (Cleanup)'},
    {'id': 'S_Wipe_2', 'label': 'Wipe Pedestals 2', 'type': 'Step', 'title': 'Action: Wipe NanoDrop pedestals after measurements'},
    {'id': 'S_Close_SW', 'label': 'Close Software', 'type': 'Software Action', 'title': 'Software Action: Close the software'},
    {'id': 'S_PowerOff_ND', 'label': 'Power Off NanoDrop', 'type': 'Step', 'title': 'Action: Power off the Nanodrop (Optional)'},
    {'id': 'S_End', 'label': 'End', 'type': 'End', 'title': 'Protocol End'},

    # Implicit locations/targets - could be nodes or just features/edge types
    # {'id': 'LOC_LowerPedestal', 'label': 'Lower Pedestal', 'type': 'Location', 'title': 'Location: NanoDrop lower pedestal'}
]

# 2. Define Edges and their types/features
#    Using 'title' for hover information on edges
edges = [
    # Initialization
    {'source': 'S_Start', 'target': 'S_On_Comp', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_On_Comp', 'target': 'N_Computer', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_On_Comp', 'target': 'S_Open_SW', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_Open_SW', 'target': 'N_Software', 'type': 'Interacts With', 'title': 'Interacts With'},
    {'source': 'S_Open_SW', 'target': 'SE_Desktop', 'type': 'Location', 'title': 'Location'},
    {'source': 'S_Open_SW', 'target': 'S_Wait_Init', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_Wait_Init', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_Wait_Init', 'target': 'ST_Init_Done', 'type': 'Waits For State', 'title': 'Waits For State'}, # Or Enables/Sequence from state to next step
    {'source': 'ST_Init_Done', 'target': 'S_Select_Opt', 'type': 'Enables/Sequence', 'title': 'State Enables Sequence'},
    {'source': 'S_Select_Opt', 'target': 'SE_NuclAcid_Opt', 'type': 'Interacts With', 'title': 'Interacts With'},
    {'source': 'S_Select_Opt', 'target': 'S_RaiseArm_1', 'type': 'Sequence', 'title': 'Sequence'},

    # Blanking
    {'source': 'S_RaiseArm_1', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_RaiseArm_1', 'target': 'ST_Arm_Raised_1', 'type': 'Changes State', 'title': 'Changes State'},
    {'source': 'ST_Arm_Raised_1', 'target': 'S_Pipette_Blank', 'type': 'Requires State', 'title': 'Requires State'},
    {'source': 'S_Pipette_Blank', 'target': 'M_BlankBuffer', 'type': 'Uses Material (1uL)', 'title': 'Uses Material (1uL)'},
    {'source': 'S_Pipette_Blank', 'target': 'D_Pipette', 'type': 'Uses Device', 'title': 'Uses Device'},
    {'source': 'S_Pipette_Blank', 'target': 'D_PipetteTips', 'type': 'Uses Device Accessory', 'title': 'Uses Device Accessory'},
    {'source': 'S_Pipette_Blank', 'target': 'S_LowerArm_1', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_LowerArm_1', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_LowerArm_1', 'target': 'ST_Arm_Lowered_1', 'type': 'Changes State', 'title': 'Changes State'},
     {'source': 'ST_Arm_Lowered_1', 'target': 'S_Click_Blank', 'type': 'Requires State', 'title': 'Requires State'},
    {'source': 'S_Click_Blank', 'target': 'SE_Blank_Button', 'type': 'Interacts With', 'title': 'Interacts With'},
    {'source': 'S_Click_Blank', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'}, # Blanking is performed on the device
    {'source': 'S_Click_Blank', 'target': 'ST_Blanked', 'type': 'Changes State', 'title': 'Changes State'},
    {'source': 'ST_Blanked', 'target': 'S_RaiseArm_2', 'type': 'Enables/Sequence', 'title': 'State Enables Sequence (Start Loop)'}, # State enables start of loop

    # Loop Structure & Sample Measurement (Simplified to show one pass and loop back option)
    {'source': 'S_RaiseArm_2', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_RaiseArm_2', 'target': 'ST_Arm_Raised_2', 'type': 'Changes State', 'title': 'Changes State'},
    {'source': 'ST_Arm_Raised_2', 'target': 'S_Wipe_1', 'type': 'Requires State', 'title': 'Requires State'},
    {'source': 'S_Wipe_1', 'target': 'D_NanoDrop', 'type': 'On Device (Pedestals)', 'title': 'On Device (Pedestals)'},
    {'source': 'S_Wipe_1', 'target': 'M_Wipe', 'type': 'Uses Material', 'title': 'Uses Material'},
    {'source': 'S_Wipe_1', 'target': 'S_Write_Label', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_Write_Label', 'target': 'M_CurrentSample', 'type': 'Input Material Info', 'title': 'Input Material Info (Label)'},
    {'source': 'S_Write_Label', 'target': 'SE_Label_Field', 'type': 'Interacts With', 'title': 'Interacts With'},
    {'source': 'S_Write_Label', 'target': 'S_Pipette_Sample', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_Pipette_Sample', 'target': 'M_CurrentSample', 'type': 'Uses Material (1uL)', 'title': 'Uses Material (1uL)'},
    {'source': 'M_CurrentSample', 'target': 'D_NanoDrop', 'type': 'Placed On Device', 'title': 'Placed On Device (Lower Pedestal)'}, # Implicit action/location
    {'source': 'S_Pipette_Sample', 'target': 'D_Pipette', 'type': 'Uses Device', 'title': 'Uses Device'},
    {'source': 'S_Pipette_Sample', 'target': 'D_PipetteTips', 'type': 'Uses Device Accessory', 'title': 'Uses Device Accessory'},
    {'source': 'S_Pipette_Sample', 'target': 'S_LowerArm_2', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_LowerArm_2', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_LowerArm_2', 'target': 'ST_Arm_Lowered_2', 'type': 'Changes State', 'title': 'Changes State'},
    {'source': 'ST_Arm_Lowered_2', 'target': 'S_Click_Measure', 'type': 'Requires State', 'title': 'Requires State'},
    {'source': 'S_Click_Measure', 'target': 'SE_Measure_Button', 'type': 'Interacts With', 'title': 'Interacts With'},
    {'source': 'S_Click_Measure', 'target': 'D_NanoDrop', 'type': 'On Device (Start Meas)', 'title': 'On Device (Starts Measurement)'},
    {'source': 'S_Click_Measure', 'target': 'S_Wait_Measure', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_Wait_Measure', 'target': 'D_NanoDrop', 'type': 'Waits For Device Process', 'title': 'Waits For Device Process'},
    {'source': 'S_Wait_Measure', 'target': 'ST_Meas_Done', 'type': 'Waits For State', 'title': 'Waits For State'},
    {'source': 'ST_Meas_Done', 'target': 'S_Save_Meas', 'type': 'Enables/Sequence', 'title': 'State Enables Sequence'},
    {'source': 'S_Save_Meas', 'target': 'M_MeasData', 'type': 'Output Data', 'title': 'Output Data'},
    {'source': 'M_CurrentSample', 'target': 'M_MeasData', 'type': 'Has Output Data', 'title': 'Has Output Data'}, # Link sample to its data

    # Loop Back/Continue Decision (Explicitly showing the options)
    # This is a decision point conceptually after S_Save_Meas
    {'source': 'S_Save_Meas', 'target': 'S_RaiseArm_2', 'type': 'Loop Back', 'title': 'Loop: More Samples Available'},
    {'source': 'S_Save_Meas', 'target': 'S_RaiseArm_3', 'type': 'Sequence (Exit Loop)', 'title': 'Sequence: No More Samples'}, # Continues to cleanup

    # Cleanup
    {'source': 'S_RaiseArm_3', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_RaiseArm_3', 'target': 'ST_Arm_Raised_3', 'type': 'Changes State', 'title': 'Changes State'},
    {'source': 'ST_Arm_Raised_3', 'target': 'S_Wipe_2', 'type': 'Requires State', 'title': 'Requires State'},
    {'source': 'S_Wipe_2', 'target': 'D_NanoDrop', 'type': 'On Device (Pedestals)', 'title': 'On Device (Pedestals)'},
    {'source': 'S_Wipe_2', 'target': 'M_Wipe', 'type': 'Uses Material', 'title': 'Uses Material'},
    {'source': 'S_Wipe_2', 'target': 'S_Close_SW', 'type': 'Sequence', 'title': 'Sequence'},
    {'source': 'S_Close_SW', 'target': 'N_Software', 'type': 'Interacts With', 'title': 'Interacts With'},
    # Optional step handling: two possible sequences
    {'source': 'S_Close_SW', 'target': 'S_PowerOff_ND', 'type': 'Sequence (Optional)', 'title': 'Sequence (Optional Step)'},
    {'source': 'S_Close_SW', 'target': 'S_End', 'type': 'Sequence (Skip Optional)', 'title': 'Sequence (Skips Optional Step)'},
    {'source': 'S_PowerOff_ND', 'target': 'D_NanoDrop', 'type': 'On Device', 'title': 'On Device'},
    {'source': 'S_PowerOff_ND', 'target': 'S_End', 'type': 'Sequence', 'title': 'Sequence'},

]

# 3. Create the NetworkX graph (optional, but good practice)
#    Then convert to pyvis network
G = nx.DiGraph() # Use DiGraph for directed edges

# Add nodes with attributes
for node_data in nodes:
    G.add_node(node_data['id'], label=node_data['label'], type=node_data['type'], title=node_data['title'])

# Add edges with attributes
for edge_data in edges:
    G.add_edge(edge_data['source'], edge_data['target'], type=edge_data['type'], title=edge_data['title'])

# 4. Create a PyVis network
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)

# Add nodes and edges from the NetworkX graph
# pyvis automatically uses 'title' for hover info and 'label' for node text
net.from_nx(G)

# You can customize node appearance based on type if desired
# For example, color nodes differently
type_colors = {
    'Start': '#4CAF50', # Green
    'End': '#F44336',   # Red
    'Step': '#2196F3',  # Blue
    'Non-action': '#FF9800', # Orange
    'Software Action': '#9C27B0', # Purple
    'Material': '#FFEB3B', # Yellow
    'Device': '#795548', # Brown
    'Software': '#00BCD4', # Cyan
    'Software Element': '#009688', # Teal
    'State': '#607D8B', # Blue Grey
    'Data Output': '#CDDC39', # Lime
    'Device Accessory': '#BDBDBD' # Grey
}

for node in net.nodes:
    node_type = G.nodes[node['id']]['type'] # Get type from NetworkX graph
    node['color'] = type_colors.get(node_type, '#9E9E9E') # Default grey

# Optional: Customize edge appearance based on type (can make graph busy)
# For simplicity, we'll rely on hover title for edge type
# You could add code here to change edge color or style based on edge['type']


# 5. Generate the interactive visualization
output_filename = "nanodrop_protocol_graph.html"
net.show(output_filename, notebook=False)

print(f"Interactive graph visualization saved to {output_filename}")
print(f"Open the file {output_filename} in your web browser to view it.")