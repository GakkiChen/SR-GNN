# Social Relationship Prediction with Graph Neural Networks

## Abstract
A graph neural network backed with visual features extracted from VGG16 fully-connected layers. Our model generates a
social relationship graph from an image then keeps developing and correcting the graph upon receiving new images from
the same album by enforcing consistency among graph nodes.

Work in collaboration with Arushi Goel PhD candidate at University of Edinburgh and Dr. Cheston Tan at ASTAR.

Targeted to improve on top of Arushi Goel's published paper:
[An End-to-End Network for Generating Social Relationship Graphs](https://openaccess.thecvf.com/content_CVPR_2019/papers/Goel_An_End-To-End_Network_for_Generating_Social_Relationship_Graphs_CVPR_2019_paper.pdf) by enabling the network to understand simple logic from data such as one
cannot have multiple fathers/mothers, a child’s father is the spouse of its mother, etc. 

## Model Architecture
![Alt text](./img/SRGNN.PNG?raw=true "SRGNN Model")
Work progression can be found at: [Progress Report](https://drive.google.com/drive/folders/1CIr-ZgA6lK8lWynhexCZnORFroZIh3vS?usp=sharing)

An illustration of the design concept is at: [Proposed SRGNN Model](https://drive.google.com/file/d/1SojzV1r4eqeK__-GQwKAFCwCU1KlVpKg/view?usp=sharing)
