# tensorgrad
 
A tensor automatic differentiation engine written from scratch in C++. No libraries, no frameworks — just `std::shared_ptr`, a graph of nodes, and the chain rule, except every node holds a whole **matrix** instead of a single number. On top of it, the goal is a character-level language model that learns to invent names, trained entirely by gradients this engine computes for itself.

This is the third in a line, and the progression is the whole point:
 
- **CNeural** — backpropagation derived by hand and written as fixed code for one specific network. Matrices, but no autograd.
- **micrograd** — an engine that works out backpropagation automatically, at runtime, for any expression you build — but one scalar per node, far too small to train anything real.
- **tensorgrad** — the same automatic engine as micrograd, but every node holds a matrix. A matrix multiply becomes a *single* node with *one* backward rule instead of thousands of scalar nodes. That's the jump from toy to usable: enough to actually train a model.
CNeural hand-coded the gradients; micrograd automated them for scalars; tensorgrad automates them for tensors — and then uses that to train a real character-level model, with no ML libraries anywhere in the stack.
 
## The idea
 
Same core as micrograd — a graph of operations walked backward with the chain rule — with one change that carries all the weight: each `Value` node stores a `Matrix data` and a `Matrix grad` instead of a single number.
 
The graph machinery is unchanged from micrograd — the `shared_ptr` edges, the topological sort, the reverse-walk backward pass, the accumulation of gradients — because none of it cares what a node holds. What changes is every operation's backward rule, which becomes matrix calculus instead of scalar arithmetic. Those are the same matrix backprop rules I derived by hand in CNeural (BP1–BP4), now attached automatically to each node the moment it's built.
