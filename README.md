# Seq2tree Max-Encoder Bi-LSTM 


In their paper from 2018 [Conneau and colleagues](https://arxiv.org/pdf/1805.01070.pdf) proved the efficiency of bidirectional lstm in storing information of multiple
levels and type, from syntax to semantic, into nD vectors. According to their results, one of the most efficient 
architecture presented is the so-called max-encoder structures, that concatenate vectors, into a 2nD vector before passing
to a decoder structure, trained with sequence to tree tasks. In this scenario, models are trained to learn the semantic tree 
structure of a given sentence/set of vectors. 

In my implementation, that focuses but includes as commented code, the max-encoder over the last-encoder structure, I have
added an intermediate layer, that brings the dimension back to n. Basic intuition and goal of this is to obtain a composed 
representation that can be mapped in the same space as the vectors from which is originated
