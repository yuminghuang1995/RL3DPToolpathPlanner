# RL-based-planner-for-3D-printing-toolpath

![](teaser.png)

This paper presents a learning based planner for computing optimized 3D printing toolpaths on prescribed graphs, the challenges of which include the varying graph structures on different models and the large scale of nodes and edges on a graph. The planner can cover different 3D printing applications by defining their corresponding reward functions and state spaces. Toolpath generation problems in wire-frame printing, continuous fiber printing, and metallic printing are selected here to demonstrate generality. The resultant toolpaths have been applied in physical experiments to verify the performance of the planner. By this planner, wire-frame models with up to 4.2k struts can be successfully printed and up to 93.3% of sharp turns on continuous fiber toolpaths can be avoided.


# Installation

**Platform:** Windows 10/11

**Environment:** python 3.9 + pyTorch 1.12

**Package:** numpy + matplotlib + networkx + scipy

![](algorithm.png)

# Usage

**Step 1:** Open a terminal and type **python main.py --model MODEL**. We provide over 10 initial models (**MODEL**) corresponding to different printing methods.

**Step 2:** Change the **MODEL** to your model. 

**Input file formats:**

The input txt files are under the **data** folder:

Number of nodes in the input graph:

Three-dimensional coordinates of nodes: **X Y Z**.

Number of edges in the input graph:

The corresponding node indexes on each edge: **head index, tail index**.

**Step 3:** Change the checkpoint. You can use the pre-trained initial checkpoint in the **checkpoint** folder (which is automatically called by the program, default to wireframe models), or you can skip the checkpoint and generate it during training. 

**Step 4:** **Enter** to start the **main.py**. 

**Step 5:** Wait for the progress bar to reach the end and finish post-processing. Check the **figure** folder to see the process and final generated graph.

**Step 6:** Finally we get the output file, which is divided into **results** (waypoint information) and **outputs** (Output to the printer).

**Output file formats:**

The result txt files are under the **results** folder:

Index of nodes in print order: first node, second node, ... , final node.

The output txt files are under the **outputs** folder:

Three-dimensional coordinates of nodes: X Y Z , and printing normal: nx ny nz.

![](manufacturing_results.png)
