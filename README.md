# ProxSkipVIP

## Details of Files

* ### FedGDAGTvsProxSkipVIP.py
  Comparison of FedGDA-GT and ProxSkipVIP algorithm on strongly-convex strongly-concave Quadratic min-max problem.

* ### ProximalOperator.py
  Functions for computing proximal operators of some typical constraints. Like projection onto [probability simplex](https://gist.github.com/mblondel/6f3b7aaad90606b98f71). 

* ### ProxSkipVIPxMatrixGame.py
  Implement ProxSkipVIP on distributed matrix game.

* ### model.py
  Define a class for data generation of distributed min-max optimization problems. Included min-max problems are
  - Quadratic Game.
  - Matrix Game (like Policemen Burglar Problem)
