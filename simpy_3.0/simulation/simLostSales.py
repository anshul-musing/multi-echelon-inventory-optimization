
"""This module simulates a multi-echelon supply chain
and calculates inventory profile (along with associated inventory 
parameters such as on-hand, inventory position, service level, etc.) 
across time

The system follows a base stock policy with a reorder point
If inventory position <= ROP, an order of the amount 
(base stock level - current inventory level) is placed by the
facility.  Essentially it's equivalent to filling up the tank.

It is assumed that any unfulfilled order is lost
The service level is estimated based on how much
of the demand was fulfilled

One of the key features is that we do not assume any 
pre-defined distribution for demand and lead time.  We follow a
data-driven distribution.  In other words, we bootstrap sample 
from the historical data in order to simulate variability in
both demand and lead time.  However, this inherently assumes
that there is no time correlation in historical demand and 
lead time, as well as the future will be similar to history

The discrete-event simulation model stands on four 
different processes:
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

Assumption:  The first node is the supply node such as
a manufacturing plant or a vendor for which we do not
track inventory, i.e., it operates at 100% service level

"""

__author__ = 'Anshul Agarwal'


import simpy
import numpy as np


"""Class for new replenishment order placed
by a stocking facility to its upstream
replenishing stocking facility.  The order object
contains order quantity and which facility is
placing that order
"""
class new_order(object):

    def __init__(self, requester, order_qty):
        self.requester = requester
        self.orderQty = order_qty


"""Stocking facility class
Each stocking location in the multi-echelon 
network is an object of this class
"""
class stocking_facility(object):
    
    # initialize the new facility object
    def __init__(self, env, node_id, is_source, initial_inv, ROP, base_stock,
                 upstream, hist_demand, default_lead_time, lead_time_delay):
        self.env = env
        self.name = "node" + str(node_id)
        self.isSource = is_source
        self.on_hand_inventory = initial_inv
        self.inventory_position = initial_inv
        self.ROP = ROP
        self.baseStock = base_stock
        self.upstream = upstream
        self.histDemand = hist_demand
        self.defaultLeadTime = default_lead_time
        self.leadTimeDelay = lead_time_delay
        self.order_q = []
        self.totalDemand = 0.0
        self.totalShipped = 0.0
        self.serviceLevel = 0.0
        self.avgOnHand = 0.0
        self.onHandMon = []
        
        # start the processes
        self.env.process(self.check_inventory())
        self.env.process(self.prepare_replenishment())
        self.env.process(self.serve_customer())
        

    # process to place replenishment order
    def check_inventory(self):
        while True:
            yield self.env.timeout(1.0)
            if self.inventory_position <= 1.05 * self.ROP:  # add 5% to avoid rounding issues
                order_qty = self.baseStock - self.on_hand_inventory
                order = new_order(self, order_qty)
                self.upstream.order_q.append(order)
                self.inventory_position += order_qty

    # process to fulfill replenishment order
    def prepare_replenishment(self):
        while True:
            if len(self.order_q) > 0:
                order = self.order_q.pop(0)

                shipment = min(order.orderQty, self.on_hand_inventory)
                if not self.isSource:
                    self.inventory_position -= shipment
                    self.on_hand_inventory -= shipment
    
                # if the order is not complete, wait for the material to appear
                # in the inventory before the complete replenishment can be sent
                remaining_order = order.orderQty - shipment
                if remaining_order:
                    while not self.on_hand_inventory >= remaining_order:
                        yield self.env.timeout(1.0)
                    if not self.isSource:
                        self.inventory_position -= remaining_order
                        self.on_hand_inventory -= remaining_order
                self.env.process(self.ship(order.orderQty, order.requester))
            else:
                yield self.env.timeout(1.0)

    # process to deliver replenishment
    def ship(self, qty, requester):
        lead_time = requester.defaultLeadTime + \
                    np.random.choice(requester.leadTimeDelay, replace=True)  # bootstrap sample lead time delay
        yield self.env.timeout(lead_time)
        requester.on_hand_inventory += qty

    # process to serve customer demand
    def serve_customer(self):
        while True:
            self.onHandMon.append(self.on_hand_inventory)
            yield self.env.timeout(1.0)
            demand = np.random.choice(self.histDemand, replace=True)  # bootstrap sample historical
            self.totalDemand += demand
            shipment = min(demand, self.on_hand_inventory)
            self.totalShipped += shipment
            self.on_hand_inventory -= shipment
            self.inventory_position -= shipment


def simulate_network(seedinit, num_nodes, network, initial_inv, ROP,
                     base_stock, demand, lead_time, lead_time_delay):

    env = simpy.Environment()  # initialize SimPy simulation instance
    np.random.seed(seedinit)

    nodes = []  # list of the objects of the storage facility class

    for i in range(num_nodes):
        if i == 0:  # then it is the first supply node, which is assumed to have infinite inventory
            s = stocking_facility(env, i, 1, initial_inv[i], ROP[i], base_stock[i],
                                  None, np.zeros(100), lead_time[i], lead_time_delay)
        else:
            # first find the upstream facility before invoking the processes
            for j in range(num_nodes):
                if network[j][i] == 1: # then j serves i
                    s = stocking_facility(env, i, 0, initial_inv[i], ROP[i], base_stock[i],
                                          nodes[j], demand[:, i - 1], lead_time[i], lead_time_delay)
                    break
        
        nodes.append(s)

    env.run(until=360)

    # find the service level of each node
    for i in range(num_nodes):
        nodes[i].serviceLevel = nodes[i].totalShipped / (nodes[i].totalDemand + 1.0e-5)

    # find the average on-hand inventory of each node
    for i in range(num_nodes):
        if i == 0: # then it is the first supply node, which is assumed to have infinite inventory
            nodes[i].avgOnHand = 0.0
        else:
            nodes[i].avgOnHand = np.mean(nodes[i].onHandMon)

    return nodes  # return the storageNode objects

