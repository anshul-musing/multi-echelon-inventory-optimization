# multi-echelon-inventory-optimization

The modules in this repository optimize inventory for a multi-echelon supply chain network.  We follow a simulation optimization approach where the multi-echelon system is simulated using a SimPy-based discrete-event simulation.  The optimization routine follows a black box approach.  It minimizes average on-hand inventory for all stocking facilities while ensuring the desired fill rate (_&beta;_ service level) is met for locations serving customers. It calls the simulation model to determine current average on-hand inventory and service level.

We compare the following three different open-source black-box optimization libraries.  Source code is provided for all three separately:

* scipy.optimize
* gp_minimize from skopt
* rbfopt

The discrete-event simulation model calculates inventory profile (along with associated inventory parameters such as on-hand, inventory position, service level, etc.) across time.  We assume that the system follows a base stock policy with a reorder point.  If inventory position <= ROP, an order of the amount (base stock level - current inventory level) is placed by the facility.  Essentially it's equivalent to filling up the tank.  For the optimization model, both base stock level and reorder point are decision variables for each stocking facility.

The two discrete-event simualtion modules differ in the following way:

* simBackorder.py:

    It is assumed that any unfulfilled order is backordered
    and is fulfilled whenever the material is available in the 
    inventory.  The service level is estimated based on how 
    late the order was fulfilled

* simLostSales.py

    It is assumed that any unfulfilled order is lost.  The 
    service level is estimated based on how much of the 
    demand was fulfilled

One of the key features is that we do not assume any pre-defined distribution for demand and lead time.  We follow a data-driven distribution.  In other words, we bootstrap sample from the historical data in order to simulate variability in both demand and lead time.  However, this inherently assumes that there is no time correlation in historical demand and lead time, as well as the future will be similar to history.

The discrete-event simulation model stands on four different processes:
1) Place replenishment order: 
    Process used by stocking locations to place replenishment
    order to upstream facilities once inventory levels reach 
    below reorder point
2) Fulfill replenishment order:
    Process used by a facility to prepare and ship the
    replenishment ordered by its downstream facility.  Once
    the order is prepared, it fires a delivery process
3) Deliver replenishment:
    Process invoked by fulfill order.  Each replenishment 
    delivery is handled by this process.  It delivers the
    replenishment after lead time, thus increasing downstream
    facility's on hand inventory
4) Customer demand:
    Basic process to deliver customer demand from each of the
    serving locations

Assumption:  The first node is the supply node such as a manufacturing plant or a vendor for which we do not track inventory, i.e., it operates at 100% service level

Our experiments demonstrate practical applicability of our approach. While we observe substantially lower inventory levels and computationally superior results from RBFOpt, depending on the problem, search strategy, and the random start states, close enough good solution can be obtained from both RBFOpt and Scikit-Optimize. The optimization results demonstrate a preference for a centralized inventory planning scheme that help with risk pooling. Moreover, with no order placement cost, the optimal solution tends to order more frequently in order to lower inventory.  

The work is published [here]

[here]: https://arxiv.org/abs/1901.00090
