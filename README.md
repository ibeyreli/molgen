# molgen
Deep Molecule Generation

Drug design can be defined as creating new organic molecules that interact with bio-molecules such as proteins in certain ways to affect their function in the biological system. It mainly involves designing molecules that possess certain physical and the chemical properties. In the case of computer aided drug design, we may illustrate molecules as graphs where  the atoms are nodes and the chemical bonds among them are edges. With such an analogy, deep generative networks (DGNs) show the potential of being strong tools for drug and molecule design. Although one needs to make use of a number heuristics to convert the traditional models that generate grid structured data, such as images, recent researches focuses more on drug design with generative models.In this project, we aim to apply an approach to implement a deep generative model to predict best planar molecule structure where the number of atoms per element and some desired properties are given as the input. This report contains the brief information about the selected data set,  methods and experiments.

### Dataset
ChEMBL is a database for small bio-active molecules curated from scientific literature. It contains 2D structures and properties of more than 2 million drug-like molecules. The aim is to use the properties of these compounds to generate their 2D structures correctly.

### Method

In this project, the planar structure of a compound will be presented as a graph consisting of a single connected component. To obtain standard output at the end of the network, the adjacency matrix of this matrix will be assumed to have a constant number of nodes. Hence, for different compounds with different number of atoms, the size of the single connected component will change while the total number of nodes will be preserved with disconnected vertices. Due to the varying maximum number of bonds that each type of atom can make, some restrictions will be enforced on the adjacency matrix.



