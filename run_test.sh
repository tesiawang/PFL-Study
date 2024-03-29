#
## ----------------- CIFAR-10 -----------------
#data_partition="noniid-skew"
#client=10
#iternum=200
#alpha=0.8
#beta=0.1
#
## ----------------- CIFAR-100 -----------------
#data_partition="noniid-skew"
#client=10
#iternum=400
#skew_class=20
#alpha=0.8
#beta=0.1
#
## ----------------- Test different objective functions -----------------
## consider the term from fully-decentralized paper
#python pfedgraph_cosine.py --weighted_initial 0 --new_objective 2
#
## only consider the first-order term in pFedGraph
#python pfedgraph_cosine.py --weighted_initial 0 --new_objective 1
#
## the oringinal approx term in pFedGraph
#python pfedgraph_cosine.py --weighted_initial 0 --new_objective 0
#
## generalization error bound from FedCollab
#python pfedgraph_cosine.py --weighted_initial 0 --new_objective 3
#
##python pfedgraph_cosine.py --weighted_initial 1 --consider_data_quantity 1
##python pfedgraph_cosine.py --weighted_initial 0 --consider_data_quantity 1
##python pfedgraph_cosine.py --weighted_initial 0 --consider_data_quantity 0


python pfedgraph_cosine.py --weighted_initial 1 --new_objective 4 --num_local_iterations 400 --comm_round 70
#python pfedgraph_cosine.py --weighted_initial 1 --new_objective 3 --num_local_iterations 400 --comm_round 60
#python pfedgraph_cosine.py --weighted_initial 1 --new_objective 0 --num_local_iterations 400 --comm_round 60

### RESULTS:
# using the new gen bound can reach the same or better performance than the original approx term
# whether to take square root of the gen bound does not matter