# RL-based-planner-for-3D-printing-toolpath

This paper presents an efficient \textit{reinforcement learning} (RL) based planner for computing optimized 3D printing toolpaths, which can work on graphs on large scales by constructing the state space on-the-fly. The planner can cover different 3D printing applications by defining their corresponding reward functions and state spaces. Toolpath generation problems in wire-frame printing, continuous fiber printing, and metallic printing are selected here to demonstrate generality. The resultant toolpaths have been applied in physical experiments to verify the performance of the planner. By this planner, wire-frame models with up to 3.3k struts can be successfully printed, up to $93.3\%$ of sharp turns on continuous fiber toolpaths can be avoided, and the thermal distortion in metallic printing can be reduced by $24.9\%$.

![](Figures/RoboDK_setting_up.png)

# Installation
Platform: Windows + pyTorch + 

# Input and output file formats:

The input txt files are under the data folder:

Number of nodes in the input graph:

Three-dimensional coordinates of nodes: X Y Z.

Number of edges in the input graph:

The corresponding node indexes on each edge: head index, tail index.

The result txt files are under the results folder:

Index of nodes in print order: first node, second node, ... , final node.

The output txt files are under the outputs folder:

Three-dimensional coordinates of nodes: X Y Z , and printing normal: nx ny nz.

The code is coming soon.
