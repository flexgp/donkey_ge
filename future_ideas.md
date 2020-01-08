# Ideas for moving forward

8 January 2020
Bryn Reinstadler
Ideas for evolutionary/coevolutionary iterated game theoretic problems.

## Testing

Currently, the testing modules available look at iterated prisoner's dilemma, hawk and dove, and symbolic regression. We could expand to look at the following types of games:

* Generalize to other symmetric or asymmetric 2x2 games with or without conflicting interests. 

In both of the symmetric games above, the conflicting interests can be summarized as:
Temptation > Coordination > Neutral > Punishment

We could re-order these parameters by creating a generalized 2x2 game that tweaks the relative payoffs of the four options and examine the equilibria in each case of iterated play. 

* Traveling salesman problem
* Just generally more complex grammars; introducing different types of agents into game theoretic play via increased grammar complexity

* [Ecotypic variation in Hawk+Dove](https://link.springer.com/article/10.1007/BF02214162) - from the code comments

## Implementation of evolution / coevolution

* More than single-point cross-over (need more complex grammar?)
* Tweaking # of elites, population size and seeing how that affects convergence to a mean fitness

## Visualisation 

* Improving current visualizations of mean fitness / general software architecture around printing results (right now only works from a single filename, doesn't function for coev)
* Adding interactive visualization elements so that a person can tweak game design through a GUI
