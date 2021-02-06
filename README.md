# FederatedLearning
Examples of federated learning on distributed systems

To run the distributed system, call python central_net.py $DEVICE_COUNT once and python user_net.py $DEVICE_NUMBER (0 to $DEVICE_COUNT - 1) $DEVICE COUNT TIMES. You should allow the user_nets to load their data before calling the central_net.



The tutorial https://nextjournal.com/gkoehler/pytorch-mnist was used to construct the base mnist network, which was later split up for federated learning.
