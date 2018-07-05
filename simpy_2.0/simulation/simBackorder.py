
"""This module simulates a multi-echelon supply chain
and calculates inventory profile (along with associated inventory 
parameters such as on-hand, inventory position, service level, etc.) 
across time

The system follows a base stock policy with a reorder point
If inventory position <= ROP, an order of the amount 
(base stock level - current inventory level) is placed by the
facility.  Essentially it's equivalent to filling up the tank.

It is assumed that any unfulfilled order is lost
The service level is estimated based on how
late the order was fulfilled

One of the key features is that we do not assume any 
pre-defined distribution for demand and lead time.  We follow a
data-driven distribution.  In other words, we bootstrap sample 
from the historical data in order to simulate variability in
both demand and lead time.  However, this inherently assumes
that there is no time correlation in historical demand and 
lead time, as well as the future will be similar to history

Assumption:  The first node is the supply node such as
a manufacturing plant or a vendor for which we do not
track inventory, i.e., it operates at 100% service level

"""

__author__ = 'Anshul Agarwal'


from SimPy.Simulation import *
import numpy as np


"""Stocking facility class
Each stocking location in the multi-echelon 
network is an object of this class
"""
class stocking_facility:

    def __init__(self, node_id, is_source, initial_inv, ROP, base_stock, hist_demand,
                 default_lead_time, lead_time_delay):
        self.name = "node" + str(node_id)
        self.isSource = is_source
        self.on_hand_inventory = initial_inv
        self.inventory_position = initial_inv
        self.ROP = ROP
        self.baseStock = base_stock
        self.histDemand = hist_demand
        self.defaultLeadTime = default_lead_time
        self.leadTimeDelay = lead_time_delay
        self.order_q = []
        self.totalDemand = 0.0
        self.totalBackOrder = 0.0
        self.totalLateSales = 0.0
        self.serviceLevel = 0.0
        self.avgOnHand = 0.0
        self.onHandMon = Monitor()


"""Class for new replenishment order placed
by a stocking facility to its upstream
replenishing stocking facility.  The order object
contains order quantity and which facility is
placing that order
"""
class new_order:

    def __init__(self, requester, order_qty):
        self.facility = requester
        self.orderQty = order_qty


"""The discrete-event simulation model stands on four 
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
"""


class place_replenishment_order(Process):

    def __init__(self, requester, supplier):
        Process.__init__(self)
        self.facility = requester
        self.upstream = supplier

    def check_inventory(self):
        while True:
            yield hold, self, 1.0
            if self.facility.inventory_position <= 1.05 * self.facility.ROP:  # add 5% to avoid rounding issues
                order_qty = self.facility.baseStock - self.facility.on_hand_inventory
                order = new_order(self.facility, order_qty)
                self.upstream.order_q.append(order)
                self.facility.inventory_position += order_qty


class fulfill_replenishment_order(Process):

    def __init__(self, w):
        Process.__init__(self)
        self.facility = w

    def prepare_replenishment(self):
        while True:
            yield waituntil, self, lambda: len(self.facility.order_q) > 0
            order = self.facility.order_q.pop(0)

            # either there's enough inventory for complete the order quantity
            # or empty out the inventory to start preparing the
            # replenishment order
            shipment = min(order.orderQty, self.facility.on_hand_inventory)
            if not self.facility.isSource:
                self.facility.inventory_position -= shipment
                self.facility.on_hand_inventory -= shipment

            # if the order is not complete, wait for the material to appear
            # in the inventory before the complete replenishment can be sent
            remaining_order = order.orderQty - shipment
            if remaining_order:
                yield waituntil, self, lambda: self.facility.on_hand_inventory >= remaining_order
                if not self.facility.isSource:
                    self.facility.inventory_position -= remaining_order
                    self.facility.on_hand_inventory -= remaining_order
            delivery = deliver_replenishment(order.orderQty, order.facility)
            activate(delivery, delivery.ship())


class deliver_replenishment(Process):

    def __init__(self, qty, requester):
        Process.__init__(self)
        self.qty = qty
        self.facility = requester

    def ship(self):
        lead_time = self.facility.defaultLeadTime + \
                    np.random.choice(self.facility.leadTimeDelay, replace=True)  # bootstrap sample lead time delay
        yield hold, self, lead_time
        self.facility.on_hand_inventory += self.qty


class customer_demand(Process):

    def __init__(self, w):
        Process.__init__(self)
        self.facility = w

    def serve_customer(self):
        while True:
            self.facility.onHandMon.observe(y=self.facility.on_hand_inventory)
            yield hold, self, 1.0
            demand = np.random.choice(self.facility.histDemand, replace=True)  # bootstrap sample historical
            self.facility.totalDemand += demand
            shipment = min(demand + self.facility.totalBackOrder, self.facility.on_hand_inventory)
            self.facility.on_hand_inventory -= shipment
            self.facility.inventory_position -= shipment
            backorder = demand - shipment
            self.facility.totalBackOrder += backorder
            self.facility.totalLateSales += max(0.0, backorder)


def simulate_network(seedinit, num_nodes, network, initial_inv, ROP,
                     base_stock, demand, lead_time, lead_time_delay):

    initialize()  # initialize SimPy simulation instance
    np.random.seed(seedinit)

    nodes = []  # list of the objects of the storage facility class

    for i in range(num_nodes):
        if i == 0:  # then it is the first supply node, which is assumed to have infinite inventory
            s = stocking_facility(i, 1, initial_inv[i], ROP[i], base_stock[i],
                                  np.zeros(100), lead_time[i], lead_time_delay)
        else:
            s = stocking_facility(i, 0, initial_inv[i], ROP[i], base_stock[i],
                                  demand[:, i - 1], lead_time[i], lead_time_delay)
        nodes.append(s)

    # activate the simulation
    for i in range(num_nodes):
        d = customer_demand(nodes[i])
        activate(d, d.serve_customer())
        f = fulfill_replenishment_order(nodes[i])
        activate(f, f.prepare_replenishment())
        for j in range(num_nodes):
            if network[i][j] == 1:
                p = place_replenishment_order(nodes[j], nodes[i])
                activate(p, p.check_inventory())

    simulate(until=360)

    # find the service level of each node
    for i in range(num_nodes):
        nodes[i].serviceLevel = 1 - nodes[i].totalLateSales / (nodes[i].totalDemand + 1.0e-5)

    # find the average on-hand inventory of each node
    for i in range(num_nodes):
        if i == 0: # then it is the first supply node, which is assumed to have infinite inventory
            nodes[i].avgOnHand = 0.0
        else:
            nodes[i].avgOnHand = np.mean(nodes[i].onHandMon.yseries())

    return nodes  # return the storageNode objects
